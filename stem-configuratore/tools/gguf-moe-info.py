#!/usr/bin/env python3
"""gguf-moe-info: analizza un file GGUF (senza dipendenze esterne) per dimensionare una cache di esperti MoE.

Mostra: parte densa vs esperti, dimensione di un esperto su disco, byte letti per token
e una stima dei token/s in funzione dell'hit rate della cache, date le bande di RAM e NVMe.

uso: python gguf-moe-info.py MODELLO.gguf [--ram-gbs 28] [--nvme-gbs 3.5] [--cache-gb 6] [--tensors]
"""

import argparse
import os
import struct
import sys

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

# tipi dei valori GGUF: formato struct per gli scalari
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
T_STRING, T_ARRAY = 8, 9


class Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        n = self.unpack("<Q")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t):
        if t in SCALAR:
            return self.unpack(SCALAR[t])
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            it = self.unpack("<I")
            n = self.unpack("<Q")
            if it in SCALAR:  # array numerici (es. vocabolario): salta senza costruire liste enormi
                size = struct.calcsize(SCALAR[it])
                if n > 64:
                    self.f.seek(n * size, os.SEEK_CUR)
                    return f"<array di {n} elementi>"
            items = [self.value(it) for _ in range(n)]
            return items if n <= 64 else f"<array di {n} elementi>"
        raise ValueError(f"tipo GGUF sconosciuto: {t}")


def read_gguf(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: non e' un file GGUF")
        version = r.unpack("<I")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(n_dims)]
            ttype = r.unpack("<I")
            offset = r.unpack("<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        align = kv.get("general.alignment", 32)
        data_start = (f.tell() + align - 1) // align * align
    file_size = os.path.getsize(path)

    # dimensione = distanza dal tensore successivo (include al massimo il padding di allineamento)
    by_off = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(by_off):
        end = by_off[i + 1]["offset"] if i + 1 < len(by_off) else file_size - data_start
        t["size"] = end - t["offset"]
        t["file_offset"] = data_start + t["offset"]
    return version, kv, tensors, file_size


def fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def analyze(path):
    """Legge il GGUF e ritorna un dizionario con la struttura MoE (usato anche da moe-trace-sim.py)."""
    version, kv, tensors, file_size = read_gguf(path)
    arch = kv.get("general.architecture", "?")

    def hp(key, default=None):
        return kv.get(f"{arch}.{key}", default)

    r = {
        "path": path, "version": version, "kv": kv, "tensors": tensors, "file_size": file_size, "arch": arch,
        "n_layer": hp("block_count", 0), "n_expert": hp("expert_count", 0), "n_used": hp("expert_used_count", 0),
        "hp": hp,
    }
    exp = [t for t in tensors if "_exps" in t["name"]]
    r["dense_size"] = sum(t["size"] for t in tensors if "_exps" not in t["name"])
    r["exp_size"] = sum(t["size"] for t in exp)
    r["is_moe"] = bool(exp and r["n_expert"] and r["n_used"])
    if not r["is_moe"]:
        return r

    # un esperto = una fetta contigua (ultima dimensione) di ciascun tensore *_exps del layer
    per_layer = {}
    for t in exp:
        per_layer.setdefault(t["name"].split(".")[1], []).append(t)
    r["exp_tensor_names"] = sorted(t["name"].split(".")[2] for t in next(iter(per_layer.values())))
    r["slice_sizes"] = sorted({t["size"] // t["dims"][-1] for t in exp})
    r["n_exp_layers"] = len(per_layer)
    r["expert_bytes"] = r["exp_size"] / (len(per_layer) * r["n_expert"])
    r["exp_per_tok"] = r["expert_bytes"] * r["n_used"] * len(per_layer)
    # di token_embd si legge una riga per token, a meno che non faccia anche da output head (pesi legati)
    r["tied"] = "output.weight" not in {t["name"] for t in tensors}
    r["dense_per_tok"] = r["dense_size"]
    if not r["tied"]:
        r["dense_per_tok"] -= sum(t["size"] for t in tensors if t["name"].startswith("token_embd"))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--ram-gbs", type=float, default=28.0, help="banda RAM misurata (GB/s)")
    ap.add_argument("--nvme-gbs", type=float, default=3.5, help="banda NVMe misurata (GB/s)")
    ap.add_argument("--cache-gb", type=float, default=None, help="RAM disponibile per la cache esperti (GB)")
    ap.add_argument("--tensors", action="store_true", help="elenca tutti i tensori")
    a = ap.parse_args()

    r = analyze(a.model)
    hp = r["hp"]
    n_expert, n_used = r["n_expert"], r["n_used"]

    print(f"file      : {a.model} ({fmt_bytes(r['file_size'])}, GGUF v{r['version']})")
    print(f"arch      : {r['arch']}  nome: {r['kv'].get('general.name', '?')}")
    print(f"layer     : {r['n_layer']}  n_embd: {hp('embedding_length')}  n_ff: {hp('feed_forward_length')}"
          f"  n_ff_exp: {hp('expert_feed_forward_length', '-')}")
    print(f"esperti   : {n_expert} per layer, {n_used} attivi per token"
          f"  (condivisi: {hp('expert_shared_count', 0)})")

    if a.tensors:
        print()
        for t in sorted(r["tensors"], key=lambda t: t["offset"]):
            tn = GGML_TYPE_NAMES.get(t["type"], f"type{t['type']}")
            print(f"  {t['name']:40s} {tn:8s} {str(t['dims']):28s} {fmt_bytes(t['size']):>10s} @ {t['file_offset']}")

    dense_size, exp_size = r["dense_size"], r["exp_size"]
    print()
    print(f"parte densa (sempre usata) : {fmt_bytes(dense_size)}")
    print(f"esperti (usati in parte)   : {fmt_bytes(exp_size)}  ({100 * exp_size / max(1, r['file_size']):.0f}% del file)")

    if not r["is_moe"]:
        print("\nmodello denso: ogni token legge tutti i pesi, una cache di esperti non serve.")
        return

    print(f"tensori *_exps per layer   : {', '.join(r['exp_tensor_names'])}")
    print(f"fetta di un esperto        : {', '.join(fmt_bytes(s) for s in r['slice_sizes'])} per tensore"
          f" -> {fmt_bytes(r['expert_bytes'])} per esperto")
    exp_per_tok, dense_per_tok = r["exp_per_tok"], r["dense_per_tok"]
    print(f"output head                : {'legato a token_embd (letto tutto a ogni token)' if r['tied'] else 'output.weight separato'}")
    print(f"byte letti per token       : ~{fmt_bytes(dense_per_tok)} densi + {fmt_bytes(exp_per_tok)} esperti")

    print(f"\nstima token/s (limite di banda; RAM {a.ram_gbs} GB/s, NVMe {a.nvme_gbs} GB/s):")
    t_ram_all = (dense_per_tok + exp_per_tok) / (a.ram_gbs * 1e9)
    print(f"  tutto in RAM               : {1 / t_ram_all:7.1f} tok/s")
    for hit in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        miss = exp_per_tok * (1 - hit)
        # caso migliore: lettura NVMe sovrapposta al calcolo; caso peggiore: in serie
        t_seq = t_ram_all + miss / (a.nvme_gbs * 1e9)
        t_ovl = max(t_ram_all, miss / (a.nvme_gbs * 1e9))
        print(f"  hit rate cache {hit:4.0%}        : {1 / t_seq:7.1f} tok/s (in serie)  {1 / t_ovl:7.1f} tok/s (sovrapposto)")

    if a.cache_gb is not None:
        budget = a.cache_gb * 1e9 - dense_size
        frac = max(0.0, min(1.0, budget / exp_size))
        print(f"\ncon {a.cache_gb} GB di RAM: parte densa fissa + {fmt_bytes(max(0, budget))} di cache"
              f" = {frac:.0%} degli esperti residenti")


if __name__ == "__main__":
    main()
