#!/usr/bin/env python3
"""stem: configuratore di llama.cpp per questo PC, guidato da benchmark (stile "soup").

  stem init   -m MODELLO.gguf [--template chat|codice|documenti|veloce] [--ram-gb 4]
              rileva la macchina, analizza il modello e crea stem.yaml (modificabile a mano)
  stem tune   misura con llama-bench le alternative e sceglie i parametri piu' veloci
              per la "richiesta tipica"; scrive stem-ottimizzato.yaml e un rapporto in risultati/
  stem show   mostra i parametri risolti (da stem.yaml, da tune o da una regola) e i comandi
  stem run    avvia la chat (llama-cli) con i parametri risolti;  argomenti extra dopo --
  stem serve  avvia il server (llama-server) con i parametri risolti

Valori "auto" in stem.yaml: li sceglie "stem tune"; senza tune si usa una regola prudente.
Valori espliciti in stem.yaml hanno sempre la precedenza e non vengono provati da tune.
Nessuna dipendenza esterna: se PyYAML e' installato viene usato, altrimenti un lettore YAML interno.
"""

import argparse
import ctypes
import datetime
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLAMA_BIN = os.path.join(ROOT, "llama.cpp", "build", "bin")
RUN_LIMITED = os.path.join(ROOT, "tools", "run-limited", "run-limited.exe")
RESULTS = os.path.join(ROOT, "risultati")
PATCHES = os.path.join(ROOT, "patches")   # presente solo nel progetto completo, che le applica alla build locale
GIB = 1024 ** 3

TEMPLATES = {
    "chat":      {"contesto": 8192,  "token_prompt": 500,  "token_risposta": 300, "cache": "f16"},
    "codice":    {"contesto": 16384, "token_prompt": 2000, "token_risposta": 600, "cache": "f16"},
    "documenti": {"contesto": 32768, "token_prompt": 8000, "token_risposta": 400, "cache": "q8_0"},
    "veloce":    {"contesto": 4096,  "token_prompt": 200,  "token_risposta": 200, "cache": "f16"},
}

# byte per elemento della KV cache, per stimarne la dimensione
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


# =========================================================================== YAML (sottoinsieme)

def _strip_comment(line):
    out, q = [], None
    for i, c in enumerate(line):
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(c)
    return "".join(out).rstrip()


def _split_commas(s):
    parts, cur, q = [], [], None
    for c in s:
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == ",":
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    parts.append("".join(cur))
    return parts


def _scalar(s):
    s = s.strip()
    if s in ("", "~", "null", "Null", "NULL"):
        return None
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_scalar(x) for x in _split_commas(inner)] if inner else []
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def _mini_yaml(text):
    lines = []
    for raw in text.splitlines():
        s = _strip_comment(raw.replace("\t", "  "))
        if s.strip():
            lines.append((len(s) - len(s.lstrip(" ")), s.strip()))
    pos = 0

    def block(indent):
        nonlocal pos
        if lines[pos][1].startswith("- ") or lines[pos][1] == "-":
            items = []
            while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("-"):
                items.append(_scalar(lines[pos][1][1:]))
                pos += 1
            return items
        d = {}
        while pos < len(lines) and lines[pos][0] == indent:
            key, sep, rest = lines[pos][1].partition(":")
            if not sep:
                raise ValueError(f"riga YAML non valida: '{lines[pos][1]}'")
            pos += 1
            if rest.strip():
                d[key.strip()] = _scalar(rest)
            elif pos < len(lines) and lines[pos][0] > indent:
                d[key.strip()] = block(lines[pos][0])
            else:
                d[key.strip()] = None
        return d

    if not lines:
        return {}
    data = block(lines[0][0])
    if pos != len(lines):
        raise ValueError(f"indentazione YAML non valida vicino a: '{lines[pos][1]}'")
    return data


def yaml_load(path):
    # utf-8-sig: alcuni editor (e PowerShell 5.1) salvano con il BOM all'inizio del file
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    try:
        import yaml  # PyYAML, se installato
        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml(text)


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    s = str(v)
    if s == "" or any(c in s for c in ":#[]{},'\"") or s.strip() != s or s.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
        return '"' + s.replace('"', "'") + '"'
    return s


def yaml_dump(d, indent=0):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(" " * indent + f"{k}:")
            out.append(yaml_dump(v, indent + 2))
        else:
            out.append(" " * indent + f"{k}: {_fmt(v)}")
    return "\n".join(out)


# =========================================================================== macchina

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram_status():
    """(totale, disponibile) in byte."""
    if sys.platform == "win32":
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys, ms.ullAvailPhys
    pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    return pages * size, os.sysconf("SC_AVPHYS_PAGES") * size


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return "sconosciuta"


def cpu_topology():
    """Core P ed E (CPU ibride Intel) e maschera del primo thread di ogni core P."""
    info = {"logici": os.cpu_count() or 1, "core_p": None, "core_e": 0, "maschera_p": None}
    if sys.platform != "win32":
        return info
    k32 = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 0 = RelationProcessorCore
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return info
    raw, off, cores = buf.raw, 0, []
    while off < size.value:
        rel, sz = struct.unpack_from("<II", raw, off)
        if rel == 0:
            eff = raw[off + 9]
            mask, group = struct.unpack_from("<QH", raw, off + 32)
            if group == 0:
                cores.append((eff, mask))
        off += sz
    if not cores:
        return info
    top = max(e for e, _ in cores)
    p = [m for e, m in cores if e == top]
    info["core_p"] = len(p)
    info["core_e"] = len(cores) - len(p)
    first = 0
    for m in p:
        first |= m & -m  # bit piu' basso: primo thread del core
    info["maschera_p"] = hex(first)
    return info


# =========================================================================== motori llama.cpp e GPU

def exe_name(name):
    return name + ".exe" if sys.platform == "win32" else name


def find_bench(path):
    """Cartella (anche una sottocartella) che contiene llama-bench, oppure None."""
    if not os.path.isdir(path):
        return None
    if os.path.exists(os.path.join(path, exe_name("llama-bench"))):
        return path
    for root, _, files in os.walk(path):
        if exe_name("llama-bench") in files:
            return root
    return None


def discover_engines():
    """Motori nella cartella di stem: la build locale (llama.cpp/build/bin) e i binari Vulkan (llama-vulkan/)."""
    out = {}
    for name, path in (("locale", LLAMA_BIN), ("vulkan", os.path.join(ROOT, "llama-vulkan"))):
        d = find_bench(path)
        if d:
            out[name] = d
    return out


def engines_from_cfg(cfg, base):
    eng = cfg.get("motori") or {}
    out = {}
    for name, d in eng.items():
        found = find_bench(os.path.normpath(os.path.join(base, str(d))))
        if found:
            out[name] = found
        else:
            print(f"stem: motore '{name}' non trovato in {d}: ignorato", file=sys.stderr)
    return out or discover_engines()


_devices_cache = {}


def list_devices(bin_dir):
    """Dispositivi visti da llama.cpp (qualsiasi backend: CUDA, Vulkan, SYCL, ...), con memoria e tipo."""
    if bin_dir in _devices_cache:
        return _devices_cache[bin_dir]
    try:
        r = subprocess.run([os.path.join(bin_dir, exe_name("llama-bench")), "--list-devices"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=120)
        text = r.stdout + "\n" + r.stderr
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    # il backend Vulkan dice se la GPU condivide la RAM di sistema (uma: 1 = integrata)
    uma = {int(m.group(1)): m.group(2) == "1" for m in re.finditer(r"ggml_vulkan: (\d+) = .*?\| uma: (\d)", text)}
    devs, listing = [], False
    for line in text.splitlines():
        if line.startswith("Available devices:"):
            listing = True
            continue
        m = re.match(r"\s+(\S+): (.*) \((\d+) MiB, (\d+) MiB free\)\s*$", line) if listing else None
        if m:
            name = m.group(1)
            mv = re.match(r"Vulkan(\d+)$", name)
            devs.append({"nome": name, "descrizione": m.group(2).strip(), "mib": int(m.group(3)),
                         "mib_liberi": int(m.group(4)), "integrata": uma.get(int(mv.group(1))) if mv else None})
    _devices_cache[bin_dir] = devs
    return devs


# =========================================================================== modello

def load_moe_info():
    path = os.path.join(ROOT, "tools", "gguf-moe-info.py")
    spec = importlib.util.spec_from_file_location("gguf_moe_info", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_facts(path):
    m = load_moe_info().analyze(path)
    hp = m["hp"]
    n_embd = hp("embedding_length", 0) or 0
    n_head = hp("attention.head_count", 0) or 0
    n_head_kv = hp("attention.head_count_kv", n_head) or n_head
    if isinstance(n_head_kv, list):  # alcuni modelli hanno un valore per layer
        n_head_kv = max(n_head_kv)
    head_k = hp("attention.key_length", n_embd // n_head if n_head else 0) or 0
    head_v = hp("attention.value_length", head_k) or head_k
    return {
        "size": m["file_size"], "arch": m["arch"], "n_layer": m["n_layer"], "is_moe": m["is_moe"],
        "n_expert": m["n_expert"], "n_used": m["n_used"],
        "kv_elems_per_token": m["n_layer"] * n_head_kv * (head_k + head_v) / 2,  # per K e per V
        "info": m,
    }


def tokenizer_signature(path):
    """Tipo, pre-tokenizer e numero di token del vocabolario: due modelli con la stessa firma possono fare da
    modello e bozza nella decodifica speculativa (llama.cpp controlla comunque la compatibilita' all'avvio)."""
    kv = load_moe_info().read_gguf(path)[1]
    return kv.get("tokenizer.ggml.model"), kv.get("tokenizer.ggml.pre"), str(kv.get("tokenizer.ggml.tokens"))


def find_draft(model):
    """Il GGUF piu' piccolo (meno di 1/4 del modello) con lo stesso vocabolario, cercato accanto al modello e in
    models/; None se non c'e'."""
    size = os.path.getsize(model)
    sig = tokenizer_signature(model)
    found = []
    for d in {os.path.dirname(os.path.abspath(model)), os.path.join(ROOT, "models")}:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if name.lower().endswith(".gguf") and os.path.abspath(p) != os.path.abspath(model) and os.path.getsize(p) < size / 4:
                try:
                    if tokenizer_signature(p) == sig:
                        found.append((os.path.getsize(p), p))
                except (OSError, ValueError, KeyError, struct.error):
                    pass
    return min(found)[1] if found else None


def kv_bytes(facts, ctx, ctk, ctv):
    e = facts["kv_elems_per_token"] * ctx
    return e * KV_BYTES.get(str(ctk), 2.0) + e * KV_BYTES.get(str(ctv), 2.0)


# =========================================================================== configurazione

def config_paths(args):
    """stem.yaml -> stem-ottimizzato.yaml, qwen3.yaml -> qwen3-ottimizzato.yaml (una configurazione per file)."""
    cfg = os.path.abspath(args.config)
    return cfg, os.path.splitext(cfg)[0] + "-ottimizzato.yaml"


def rel(path, base):
    """Percorso relativo a base se e' vicino (al massimo una risalita), altrimenti assoluto."""
    try:
        r = os.path.relpath(path, base).replace("\\", "/")
        return r if not r.startswith("../..") else os.path.abspath(path).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def resolve(cfg_path, tuned_path, need_model=True):
    """Ritorna (parametri risolti, origine di ogni parametro, contesto)."""
    if not os.path.exists(cfg_path):
        sys.exit(f"stem: {cfg_path} non esiste: crearlo con 'stem init -m MODELLO.gguf'")
    cfg = yaml_load(cfg_path)
    src = os.path.basename(cfg_path)
    base = os.path.dirname(cfg_path)
    model = os.path.normpath(os.path.join(base, str(cfg.get("modello", ""))))
    if need_model and not os.path.exists(model):
        sys.exit(f"stem: modello non trovato: {model}")
    llama = cfg.get("llama") or {}
    mem = cfg.get("memoria") or {}
    req = cfg.get("richiesta_tipica") or {}
    machine = cfg.get("macchina") or {}

    tuned = {}
    if os.path.exists(tuned_path):
        t = yaml_load(tuned_path)
        same = t.get("modello") == cfg.get("modello") and t.get("modello_byte") == os.path.getsize(model) \
            and t.get("ram_gb") == mem.get("ram_gb") and t.get("contesto") == llama.get("contesto")
        if same:
            tuned = t.get("valori") or {}
        else:
            print(f"stem: {os.path.basename(tuned_path)} riguarda un'altra configurazione (modello, RAM o contesto):"
                  " ignorato, rilanciare 'stem tune'", file=sys.stderr)

    facts = model_facts(model)
    total, avail = ram_status()
    ram_gb = float(mem.get("ram_gb", 0) or 0)
    ctx = int(llama.get("contesto") or 8192)
    ctk, ctv = llama.get("cache_k") or "f16", llama.get("cache_v") or "f16"
    need = facts["size"] + kv_bytes(facts, ctx, ctk, ctv) + 0.5 * GIB
    limit = ram_gb * GIB if ram_gb > 0 else avail
    big = need > limit

    n_p = machine.get("core_p") or cpu_topology()["core_p"] or max(1, (os.cpu_count() or 2) // 2)
    rules = {
        "thread": n_p, "thread_prompt": n_p, "batch": 2048, "ubatch": 512, "flash_attn": "auto",
        "repack": not big, "poll": 50, "solo_core_p": False,
    }

    params, origin = {}, {}
    for k, rule in rules.items():
        v = llama.get(k, "auto")
        if v != "auto" and v is not None:
            params[k], origin[k] = v, src
        elif k in tuned:
            params[k], origin[k] = tuned[k], "tune"
        else:
            params[k], origin[k] = rule, "regola"

    # motore (cartella llama.cpp) e uso della GPU
    engines = engines_from_cfg(cfg, base)
    if not engines:
        sys.exit("stem: nessun motore llama.cpp trovato: compilarlo o indicarne la cartella in 'motori' (vedi README)")
    gpu = cfg.get("gpu") or {}
    want = cfg.get("motore", "auto")
    if want not in (None, "auto"):
        if want not in engines:
            sys.exit(f"stem: motore '{want}' non disponibile (motori: {', '.join(engines)})")
        params["motore"], origin["motore"] = want, src
    elif tuned.get("motore") in engines:
        params["motore"], origin["motore"] = tuned["motore"], "tune"
    else:
        params["motore"], origin["motore"] = ("locale" if "locale" in engines else next(iter(engines))), "regola"
    devices = list_devices(engines[params["motore"]])
    integrated_only = bool(devices) and all(d["integrata"] for d in devices)
    # regola senza tune: GPU dedicata -> fit automatico di llama.cpp; GPU integrata o nessuna -> solo CPU
    gpu_rules = {"dispositivi": "auto", "strati": 0, "esperti_su_cpu": 0, "fit": False, "op_offload": False} \
        if not devices or integrated_only else \
        {"dispositivi": "auto", "strati": "auto", "esperti_su_cpu": 0, "fit": True, "op_offload": True}
    for k, rule in gpu_rules.items():
        v = gpu.get(k, "auto")
        if v != "auto" and v is not None:
            params[k], origin[k] = v, src
        elif k in tuned and tuned.get("motore") == params["motore"]:
            params[k], origin[k] = tuned[k], "tune"
        else:
            params[k], origin[k] = rule, "regola"
    if big and params["repack"] is True:
        print("stem: il modello non sta nel limite di RAM: repack disattivato (i pesi copiati finirebbero nel file di paging)",
              file=sys.stderr)
        params["repack"], origin["repack"] = False, "regola (modello grande)"
    for k, default in (("contesto", 8192), ("cache_k", "f16"), ("cache_v", "f16")):
        params[k], origin[k] = (llama.get(k), src) if llama.get(k) is not None else (default, "regola")

    # decodifica speculativa: senza tune resta spenta (regola prudente)
    spec = cfg.get("speculativa") or {}
    draft = spec.get("modello_bozza")
    draft_path = None
    if draft not in (None, "none", "auto"):
        draft_path = os.path.normpath(os.path.join(base, str(draft)))
        if not os.path.exists(draft_path):
            print(f"stem: modello di bozza non trovato ({draft_path}): decodifica speculativa con bozza esclusa", file=sys.stderr)
            draft_path = None
    for k, yk, rule in (("spec_tipo", "tipo", "none"), ("spec_n", "token_proposti", 4)):
        v = spec.get(yk, "auto")
        if v != "auto" and v is not None:
            params[k], origin[k] = v, src
        elif k in tuned:
            params[k], origin[k] = tuned[k], "tune"
        else:
            params[k], origin[k] = rule, "regola"
    if params["spec_tipo"] == "draft-simple" and not draft_path:
        params["spec_tipo"], origin["spec_tipo"] = "none", "regola (manca la bozza)"

    mask = machine.get("maschera_p")
    if isinstance(mask, int):  # PyYAML legge 0x... come numero
        mask = hex(mask)

    ctx_info = {
        "cfg": cfg, "model": model, "facts": facts, "big": big, "need": need, "limit": limit,
        "ram_gb": ram_gb, "priority": mem.get("priorita", "below"), "req": req, "maschera_p": mask,
        "server": cfg.get("server") or {}, "tune": cfg.get("tune") or {}, "explicit": {k for k, o in origin.items() if o == src},
        "engines": engines, "fit_margin": int(gpu.get("margine_mib", 1024) or 1024), "draft": draft_path, "base": base,
    }
    return params, origin, ctx_info


def spec_args(p, ctx):
    """Argomenti della decodifica speculativa (validi per llama-cli e llama-server)."""
    t = p.get("spec_tipo", "none")
    if t in (None, "none"):
        return []
    # il tipo va indicato: con -md da solo llama.cpp lo ricava dai metadati della bozza e resta "none"
    a = ["--spec-type", str(t)]
    if t == "draft-simple" and ctx.get("draft"):
        a += ["-md", ctx["draft"], "--spec-draft-n-max", str(p["spec_n"])]
    return a


def gpu_args(p, ctx, bench):
    """Argomenti GPU per llama-bench (bench=True) o per llama-cli/llama-server; nessuno se il motore non ha GPU."""
    if not list_devices(ctx["engines"][p["motore"]]):
        return []
    a = []
    if p["dispositivi"] not in (None, "auto"):
        a += ["-dev", str(p["dispositivi"])]
    if p["fit"] is True:
        fit = ["-fitt", str(ctx["fit_margin"]), "-fitc", str(p["contesto"])]
        a += fit if bench else ["-fit", "on"] + fit
    else:
        if not bench:
            a += ["-fit", "off"]
        if p["strati"] not in (None, "auto"):
            a += ["-ngl", str(p["strati"])]
        if p["esperti_su_cpu"]:
            a += ["-ncmoe", str(p["esperti_su_cpu"])]
    if p["op_offload"] is False:
        a += ["-nopo", "1"] if bench else ["--no-op-offload"]
    return a


def fa_arg(v):
    if v is True or v == "on":
        return "on"
    if v is False or v == "off":
        return "off"
    return "auto"


def llama_args(p, ctx):
    a = ["-m", ctx["model"], "-c", str(p["contesto"]), "-t", str(p["thread"]), "-tb", str(p["thread_prompt"]),
         "-b", str(p["batch"]), "-ub", str(p["ubatch"]), "-fa", fa_arg(p["flash_attn"]),
         "-ctk", str(p["cache_k"]), "-ctv", str(p["cache_v"]), "--poll", str(p["poll"])]
    if p["repack"] is False:
        a.append("-nr")
    if p["solo_core_p"] is True and ctx["maschera_p"]:
        a += ["-C", ctx["maschera_p"], "--cpu-strict", "1"]
    return a + gpu_args(p, ctx, bench=False) + spec_args(p, ctx)


def limiter(ctx, log_every=0):
    if ctx["ram_gb"] <= 0:
        return []
    if not os.path.exists(RUN_LIMITED):
        sys.exit("stem: run-limited.exe mancante: compilarlo (vedi README) oppure usare memoria.ram_gb: 0")
    return [RUN_LIMITED, "--max-ws-mb", str(int(ctx["ram_gb"] * 1024)), "--priority", str(ctx["priority"]),
            "--log-every-s", str(log_every), "--"]


def exe(name, bin_dir=LLAMA_BIN):
    path = os.path.join(bin_dir, exe_name(name))
    if not os.path.exists(path):
        sys.exit(f"stem: {path} mancante (vedi README, sezione installazione)")
    return path


def quote(a):
    return f'"{a}"' if " " in a else a


# =========================================================================== init

def cmd_init(args):
    cfg_path, _ = config_paths(args)
    if os.path.exists(cfg_path) and not args.forza:
        sys.exit(f"stem: {cfg_path} esiste gia' (usare --forza per sovrascriverlo)")
    model = os.path.abspath(args.modello)
    if not os.path.exists(model):
        sys.exit(f"stem: modello non trovato: {model}")
    tpl = TEMPLATES[args.template]
    facts = model_facts(model)
    topo = cpu_topology()
    total, avail = ram_status()
    base = os.path.dirname(cfg_path)
    cache = tpl["cache"]
    # la V cache quantizzata richiede flash attention
    fa = "true" if cache != "f16" else "auto"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    moe = (f"MoE, {facts['n_expert']} esperti di cui {facts['n_used']} attivi" if facts["is_moe"] else "denso")
    engines = {}
    for d in args.llama or []:
        found = find_bench(os.path.abspath(d))
        if not found:
            sys.exit(f"stem: in {d} non c'e' llama-bench (servono llama-bench, llama-cli e llama-server)")
        engines[f"llama{len(engines) + 1}" if len(args.llama) > 1 else "llama"] = found
    engines.update({k: v for k, v in discover_engines().items() if v not in engines.values()})
    if not engines:
        sys.exit("stem: nessun motore llama.cpp trovato: indicare la cartella dei programmi con --llama (vedi README)")
    eng_lines, gpus = [], []
    for name, d in engines.items():
        devs = list_devices(d)
        desc = ", ".join(f"{x['nome']} {x['descrizione']}" for x in devs) or "solo CPU"
        eng_lines.append(f"  {name}: {rel(d, base)}   # dispositivi: {desc}")
        for x in devs:
            tipo = "integrata" if x["integrata"] else "dedicata" if x["integrata"] is False else "?"
            gpus.append(f"{x['nome']} {x['descrizione']} ({x['mib']} MiB, {tipo})")
    engines_yaml = "\n".join(eng_lines)
    gpus_yaml = _fmt(gpus) if gpus else "[]"
    draft = find_draft(model)
    draft_yaml = rel(draft, base) if draft else "none"
    prompt_default = os.path.join(ROOT, "tools", "moe-trace", "prompt-tecnico-en.txt")
    prompt_yaml = rel(prompt_default, base) if os.path.exists(prompt_default) else "none"
    text = f"""# stem.yaml - configurazione di llama.cpp per questo PC
# Creato da "stem init" il {now}. Si puo' modificare a mano.
#   stem tune    misura le alternative e sceglie i valori "auto" piu' veloci
#   stem show    mostra i parametri risolti e i comandi
#   stem run     chat (llama-cli)          stem serve   server (llama-server)

modello: {rel(model, base)}   # {facts['arch']}, {facts['size'] / 1e9:.2f} GB, {facts['n_layer']} layer, {moe}
template: {args.template}                 # chat | codice | documenti | veloce

# "stem tune" sceglie i parametri che completano questa richiesta nel minor tempo
richiesta_tipica:
  token_prompt: {tpl['token_prompt']}
  token_risposta: {tpl['token_risposta']}

memoria:
  ram_gb: {args.ram_gb:g}                   # RAM massima per llama.cpp (tetto rigido); 0 = nessun limite
  priorita: below               # priorita' del processo: idle | below | normal

# parametri di llama.cpp: "auto" = scelto da "stem tune" (senza tune: regola prudente)
llama:
  contesto: {tpl['contesto']}               # token di contesto (-c)
  thread: auto                  # thread per la generazione (-t)
  thread_prompt: auto           # thread per il prompt (-tb)
  batch: auto                   # -b
  ubatch: auto                  # -ub
  flash_attn: {fa}               # true | false | auto (-fa)
  cache_k: {cache}                 # tipo della KV cache: f16 | q8_0 | q4_0 (-ctk)
  cache_v: {cache}                 # -ctv; se non e' f16 serve flash_attn: true
  repack: auto                  # true | false | auto; con modelli piu' grandi della RAM e' sempre false (-nr)
  poll: auto                    # 0..100: attesa attiva dei thread (--poll)
  solo_core_p: auto             # true = un thread per core P, fissati (-C ... --cpu-strict 1)

# motori llama.cpp: cartelle con llama-bench, llama-cli e llama-server (anche build CUDA, Vulkan, SYCL, ...)
motori:
{engines_yaml}
motore: auto                    # nome di un motore qui sopra | auto = scelto da "stem tune"

# GPU (qualsiasi backend): "auto" = scelto da "stem tune" misurando le alternative
gpu:
  dispositivi: auto             # -dev: es. CUDA0, Vulkan0 | none = nessuna GPU | auto
  strati: auto                  # -ngl: layer nella memoria della GPU (0 = nessuno, 999 = tutti) | auto
  esperti_su_cpu: auto          # -ncmoe: esperti MoE dei primi N layer restano in RAM | auto
  fit: auto                     # true = llama.cpp adatta da solo layer ed esperti alla memoria GPU (-fit)
  op_offload: auto              # true = la GPU calcola anche il prompt con pesi in RAM | false (--no-op-offload)
  margine_mib: 1024             # memoria GPU da lasciare libera quando fit e' attivo (-fitt)

# decodifica speculativa: un modello piccolo ("bozza") propone alcuni token e il modello grande li verifica
# in un solo passo; conviene solo se la bozza indovina spesso e gli esperti non arrivano dal disco
speculativa:
  tipo: auto                    # none | draft-simple (usa modello_bozza) | ngram-simple (ripetizioni nel testo) | auto
  modello_bozza: {draft_yaml}   # GGUF piccolo con lo stesso vocabolario del modello | none
  token_proposti: auto          # token proposti per passo (--spec-draft-n-max) | auto

server:
  host: 127.0.0.1
  porta: 8080
  parallelo: 1                  # conversazioni servite insieme (-np); se il modello non sta in RAM rallenta tutto

# alternative provate da "stem tune"
tune:
  ripetizioni: 2                # ripetizioni di ogni misura in un giro
  giri: 2                       # giri a ordine alternato (riducono l'effetto delle variazioni del PC)
  soglia_pct: 3                 # si cambia un valore solo se e' piu' veloce almeno di questa percentuale
  priorita: normal              # priorita' durante le misure (il tetto di RAM resta attivo)
  thread: [4, 6, 8, 10]
  ubatch: [256, 512]
  poll: [0, 50]
  spec_token: [2, 4, 8]         # token proposti provati per la decodifica speculativa (serve modello_bozza)
  spec_ngram: false             # true = prova anche ngram-simple (utile con testi ripetitivi, es. codice)
  prompt: {prompt_yaml}   # testo per le misure con llama-server (decodifica speculativa)

# rilevato da "stem init" (informativo; core_p e maschera_p sono usati da tune)
macchina:
  cpu: {_fmt(cpu_name())}
  thread_logici: {topo['logici']}
  core_p: {topo['core_p'] if topo['core_p'] else 'null'}
  core_e: {topo['core_e']}
  maschera_p: {('"' + topo['maschera_p'] + '"') if topo['maschera_p'] else 'null'}
  ram_totale_gb: {total / GIB:.1f}
  gpu: {gpus_yaml}
"""
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"stem: creato {cfg_path}")
    print(f"  modello : {rel(model, base)} ({facts['size'] / 1e9:.2f} GB, {moe})")
    print(f"  CPU     : {cpu_name()} - core P {topo['core_p']}, core E {topo['core_e']}, thread logici {topo['logici']}")
    print(f"  RAM     : totale {total / GIB:.1f} GiB, disponibile ora {avail / GIB:.1f} GiB, limite scelto {args.ram_gb:g} GB")
    print(f"  motori  : {', '.join(engines)}")
    print(f"  GPU     : {'; '.join(gpus) if gpus else 'nessuna vista da llama.cpp'}")
    print(f"  bozza   : {draft_yaml} (decodifica speculativa)")
    print("Prossimo passo: 'stem tune' per misurare i parametri migliori.")


# =========================================================================== tune

def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url, data=None, timeout=3600):
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # niente proxy per 127.0.0.1
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def server_measure(b, p, prompt, n_predict, reps, report):
    """Avvia llama-server con i parametri p (tetto di RAM compreso), fa un riscaldamento e reps richieste con il
    prompt; ritorna i campioni di token/s (prompt, risposta) e le bozze accettate. Serve per cio' che llama-bench
    non sa misurare, come la decodifica speculativa."""
    port = free_port()
    cmd = limiter(b.ctx) + [exe("llama-server", b.ctx["engines"][p["motore"]])] + llama_args(p, b.ctx) + \
        ["-np", "1", "--host", "127.0.0.1", "--port", str(port)]
    report.append("$ " + " ".join(quote(x) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    pp, tg, drafted, accepted = [], [], 0, 0
    try:
        t0 = datetime.datetime.now()
        while True:
            try:
                if http_json(url + "/health", timeout=2).get("status") == "ok":
                    break
            except OSError:
                pass
            if proc.poll() is not None or (datetime.datetime.now() - t0).total_seconds() > 300:
                report.append("    (il server non e' partito)")
                return None
            time.sleep(0.5)
        req = {"prompt": prompt, "temperature": 0.8, "seed": 42, "cache_prompt": False}
        http_json(url + "/completion", dict(req, n_predict=8))  # riscaldamento
        for i in range(reps):
            t = http_json(url + "/completion", dict(req, n_predict=n_predict, seed=42 + i))["timings"]
            pp.append(t["prompt_per_second"])
            tg.append(t["predicted_per_second"])
            drafted += t.get("draft_n", 0)
            accepted += t.get("draft_n_accepted", 0)
            report.append(f"    prompt {t['prompt_per_second']:.2f}, risposta {t['predicted_per_second']:.2f} token/s,"
                          f" bozze {t.get('draft_n_accepted', 0)}/{t.get('draft_n', 0)}")
    finally:
        # run-limited avvia llama-server come figlio: si chiude l'intero albero di processi
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"] if sys.platform == "win32" else ["kill", str(proc.pid)],
                       capture_output=True)
        proc.wait()
    return pp, tg, drafted, accepted


def stats_of(samples):
    m = sum(samples) / len(samples)
    sd = (sum((x - m) ** 2 for x in samples) / (len(samples) - 1)) ** 0.5 if len(samples) > 1 else 0.05 * m
    return m, sd / len(samples) ** 0.5, len(samples)


class Bench:
    def __init__(self, params, ctx, report):
        self.p = params
        self.ctx = ctx
        self.report = report
        tcfg = ctx["tune"]
        self.reps = int(tcfg.get("ripetizioni", 2))
        self.rounds = int(tcfg.get("giri", 2))
        req = ctx["req"]
        self.req_p = int(req.get("token_prompt", 500))
        self.req_n = int(req.get("token_risposta", 300))
        big = ctx["big"]
        # prompt alla lunghezza della richiesta tipica (max 512): con i MoE piu' grandi della RAM ogni blocco
        # di prompt rilegge quasi tutti gli esperti, quindi un prompt corto sottostima molto la velocita'
        self.np = min(self.req_p, 512)
        self.ng = 32 if big else 64
        if big:
            # misure lente: una ripetizione per giro, ma almeno 2 giri a ordine alternato
            self.reps = 1
            self.rounds = max(2, self.rounds)
        # durante le misure la priorita' e' normale (configurabile): con "below" ogni altra attivita' falsa i numeri
        self.ctx = dict(ctx, priority=tcfg.get("priorita", "normal"))

    def base_args(self, threads, overrides):
        p = dict(self.p)
        p.update(overrides)
        a = ["-m", self.ctx["model"], "-o", "jsonl", "-r", str(self.reps), "-t", str(threads),
             "-b", str(p["batch"]), "-ub", str(p["ubatch"]), "-fa", fa_arg(p["flash_attn"]),
             "-ctk", str(p["cache_k"]), "-ctv", str(p["cache_v"]), "--poll", str(p["poll"]),
             "--repack", "1" if p["repack"] is not False else "0"]
        if p["solo_core_p"] is True and self.ctx["maschera_p"]:
            a += ["-C", self.ctx["maschera_p"], "--cpu-strict", "1"]
        return a + gpu_args(p, self.ctx, bench=True)

    def run(self, args, engine):
        cmd = limiter(self.ctx) + [exe("llama-bench", self.ctx["engines"][engine])] + args
        self.report.append("$ " + " ".join(quote(x) for x in cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rows = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for row in rows:
            kind = "prompt" if row["n_prompt"] else "risposta"
            self.report.append(f"    {kind:8s} t={row['n_threads']:<3d} ub={row['n_ubatch']:<5d} fa={row['flash_attn']:<3d}"
                               f" poll={row['poll']:<4d} repack={int(row['repack'])} mask={row['cpu_mask']}"
                               f" ngl={row.get('n_gpu_layers')} ncmoe={row.get('n_cpu_moe')} fit={row.get('fit_target')}"
                               f" nopo={row.get('no_op_offload')} [{row.get('backends')}] -> {row['avg_ts']:8.2f} token/s")
        if r.returncode != 0 and not rows:
            self.report.append("    (fallito) " + " | ".join(r.stderr.strip().splitlines()[-3:]))
        return rows

    def measure(self, which, threads, overrides, flag=None, values=None):
        """Misura prompt ('pp') o risposta ('tg'); con flag/values prova piu' valori.
        Ripete per piu' giri invertendo l'ordine dei valori, cosi' le variazioni del PC pesano su tutti."""
        rows = []
        for g in range(self.rounds):
            rows += self.measure_once(which, threads, overrides, flag, values, reverse=g % 2 == 1)
        return rows

    def measure_once(self, which, threads, overrides, flag=None, values=None, reverse=False):
        a = self.base_args(threads, overrides)
        a += ["-p", str(self.np), "-n", "0"] if which == "pp" else ["-p", "0", "-n", str(self.ng)]
        if flag:
            # rimuove il valore base del parametro provato e mette la lista
            if flag in a:
                i = a.index(flag)
                del a[i:i + 2]
            vals = list(reversed(values)) if reverse else list(values)
            a += [flag, ",".join(str(v) for v in vals)]
        return self.run(a, overrides.get("motore", self.p["motore"]))

    def request_time(self, pp, tg):
        return self.req_p / pp + self.req_n / tg

    def request_time_se(self, pp, tg):
        """(tempo della richiesta tipica, errore standard, campioni) da (media, errore, campioni) di prompt e risposta."""
        mp, sp, np_ = pp
        mg, sg, ng_ = tg
        t = self.req_p / mp + self.req_n / mg
        se = ((self.req_p * sp / mp ** 2) ** 2 + (self.req_n * sg / mg ** 2) ** 2) ** 0.5
        return t, se, min(np_, ng_)


def cmd_tune(args):
    cfg_path, tuned_path = config_paths(args)
    params, origin, ctx = resolve(cfg_path, "")  # parte dalle regole, ignora tune precedenti
    explicit = ctx["explicit"]
    tcfg = ctx["tune"]
    report = []
    b = Bench(params, ctx, report)
    t0 = datetime.datetime.now()

    def log(msg):
        print(msg)
        report.append(msg)

    log(f"# stem tune - {t0:%Y-%m-%d %H:%M}")
    log(f"modello: {ctx['model']}")
    log(f"richiesta tipica: {b.req_p} token di prompt + {b.req_n} di risposta;"
        f" misure da {b.np} (prompt) e {b.ng} (risposta) token, {b.reps} ripetizioni")
    if ctx["big"]:
        log(f"modello piu' grande del limite di RAM ({ctx['need'] / GIB:.1f} GiB necessari, limite {ctx['limit'] / GIB:.1f}):"
            " repack disattivato, tetto di memoria attivo, misure ridotte")
    gpu_keys = ("motore", "dispositivi", "strati", "esperti_su_cpu", "fit", "op_offload")
    cur = {k: params[k] for k in ("thread", "thread_prompt", "batch", "ubatch", "flash_attn", "repack", "poll",
                                  "solo_core_p", "spec_tipo", "spec_n") + gpu_keys}

    def best_of(rows, field, norm=lambda x: x):
        """valore -> (media token/s, errore standard della media, campioni) su tutte le ripetizioni e i giri."""
        by = {}
        for r in rows:
            by.setdefault(norm(r[field]), []).extend(r.get("samples_ts") or [r["avg_ts"]])
        out = {}
        for k, s in by.items():
            m = sum(s) / len(s)
            sd = (sum((x - m) ** 2 for x in s) / (len(s) - 1)) ** 0.5 if len(s) > 1 else 0.05 * m
            out[k] = (m, sd / len(s) ** 0.5, len(s))
        return out

    noise = []  # incertezza relativa delle misure, per l'avviso finale

    def fmt_ts(v):
        return f"{v[0]:.1f}±{v[1]:.1f}"

    n_p = (ctx["cfg"].get("macchina") or {}).get("core_p")
    max_t = os.cpu_count() or 16
    threads = sorted({int(t) for t in tcfg.get("thread", [4, 6, 8, 10]) if 0 < int(t) <= max_t} | ({n_p} if n_p else set()))
    if ctx["big"]:
        threads = sorted({t for t in threads if t <= (n_p or 6)})[-2:] or threads[:2]

    # un valore cambia rispetto a quello attuale solo se il vantaggio supera la soglia (il resto e' rumore)
    soglia = float(tcfg.get("soglia_pct", 3)) / 100
    log(f"si cambia un valore solo con almeno il {soglia:.0%} di vantaggio; {b.rounds} giri a ordine alternato,"
        f" priorita' {b.ctx['priority']}")

    def choose(scores, current, higher_is_better, name=str):
        """scores: valore -> (punteggio, errore standard). Si cambia solo con un vantaggio oltre la soglia
        e almeno 2 volte l'incertezza della differenza. Ritorna (scelta, motivo)."""
        for m, se, n in scores.values():
            if m > 0 and n > 1:
                noise.append(se / m)
        best = (max if higher_is_better else min)(scores, key=lambda k: scores[k][0])
        if best == current or current not in scores:
            return best, f"scelto {name(best)}"
        (mb, sb, nb), (mc, sc, nc) = scores[best], scores[current]
        if min(nb, nc) < 2:
            return current, f"una sola misura: dati insufficienti per cambiare, resta {name(current)}"
        gain = mb / mc - 1 if higher_is_better else 1 - mb / mc
        z = abs(mb - mc) / max((sb ** 2 + sc ** 2) ** 0.5, 1e-9)
        if gain < soglia:
            return current, f"vantaggio di {name(best)} solo {gain:.1%}: resta {name(current)}"
        if z < 2:
            return current, f"vantaggio di {name(best)} {gain:.1%} ma dentro il rumore delle misure: resta {name(current)}"
        return best, f"scelto {name(best)} (+{gain:.1%})"

    # 1-3) parametri che toccano solo la risposta o solo il prompt: si massimizzano i token/s
    def single(title, key, which, threads_key, flag, values, field):
        if key in explicit:
            return
        log(f"\n== {title}")
        res = best_of(b.measure(which, cur[threads_key], cur, flag, values), field)
        if not res:
            log("   nessuna misura valida, resta " + str(cur[key]))
            return
        cur[key], why = choose(res, cur[key], True)
        log("   " + ", ".join(f"{k}: {fmt_ts(v)}" for k, v in sorted(res.items())) + f" token/s -> {why}")

    # 0) motore e GPU: ogni configurazione e' un insieme di valori; si misurano tutte con la stessa regola
    def gpu_candidates():
        facts = ctx["facts"]
        n = int(facts["n_layer"])
        cands = []
        for name, d in ctx["engines"].items():
            if ctx["big"] and name != "locale" and "locale" in ctx["engines"] and os.path.isdir(PATCHES):
                log(f"   motore {name}: escluso, il modello non sta in RAM e solo il motore locale ha le patch per questo caso")
                continue
            devs = list_devices(d)
            cands.append((f"{name}: solo CPU", dict(motore=name, strati=0, esperti_su_cpu=0, fit=False, op_offload=False)))
            if not devs:
                continue
            if ctx["big"] and all(x["integrata"] for x in devs):
                log(f"   motore {name}: GPU integrata esclusa, condivide la RAM che il modello gia' non ha")
                continue
            vram = sum(x["mib_liberi"] for x in devs) * 1024 ** 2
            gpu = dict(motore=name, esperti_su_cpu=0, fit=False, op_offload=True)
            if facts["size"] < 0.9 * vram:
                cands.append((f"{name}: tutto sulla GPU", dict(gpu, strati=999)))
            cands.append((f"{name}: meta' dei layer sulla GPU", dict(gpu, strati=n // 2)))
            if facts["is_moe"]:
                cands.append((f"{name}: GPU tranne gli esperti (in RAM)", dict(gpu, strati=999, esperti_su_cpu=n)))
            cands.append((f"{name}: fit automatico di llama.cpp", dict(gpu, strati="auto", fit=True)))
            cands.append((f"{name}: GPU solo per il prompt", dict(gpu, strati=0)))
        return cands

    if ctx["big"]:
        # con un modello piu' grande della RAM la cache di Windows si riempie misura dopo misura:
        # un primo giro a vuoto evita di favorire le configurazioni misurate per ultime
        log("\n== riscaldamento della cache (non conta)")
        b.measure_once("tg", cur["thread"], cur)

    if not (set(gpu_keys) & explicit):
        log("\n== 0. motore e GPU")
        cands = gpu_candidates()
        if len(cands) > 1:
            samples = {label: {"pp": [], "tg": []} for label, _ in cands}
            for g in range(b.rounds):
                order = cands if g % 2 == 0 else list(reversed(cands))
                for label, over in order:
                    o = dict(cur, **over)
                    samples[label]["pp"] += b.measure_once("pp", cur["thread_prompt"], o)
                    samples[label]["tg"] += b.measure_once("tg", cur["thread"], o)
            times, desc = {}, {}
            for label, _ in cands:
                pp = best_of(samples[label]["pp"], "n_prompt").get(b.np)
                tg = best_of(samples[label]["tg"], "n_gen").get(b.ng)
                if pp and tg and pp[0] > 0 and tg[0] > 0:
                    times[label] = b.request_time_se(pp, tg)
                    log(f"   {label}: prompt {fmt_ts(pp)}, risposta {fmt_ts(tg)} token/s,"
                        f" richiesta tipica {times[label][0]:.2f}±{times[label][1]:.2f} s")
                else:
                    log(f"   {label}: misura fallita (vedi rapporto)")
            if times:
                base_label = cands[0][0] if cands[0][0] in times else None
                best, why = choose(times, base_label, False)
                cur.update(dict(cands)[best])
                log(f"   -> {why}")
        else:
            log(f"   una sola configurazione possibile: {cands[0][0] if cands else 'solo CPU, motore ' + cur['motore']}")

    single("1. thread per la risposta", "thread", "tg", "thread", "-t", threads, "n_threads")
    single("2. thread per il prompt", "thread_prompt", "pp", "thread_prompt", "-t", threads, "n_threads")
    if not ctx["big"]:
        single("3. ubatch per il prompt", "ubatch", "pp", "thread_prompt", "-ub", [int(u) for u in tcfg.get("ubatch", [256, 512])], "n_ubatch")
        cur["batch"] = max(int(cur["ubatch"]), int(cur["batch"]))

    # 4-6) opzioni che toccano prompt e risposta: si minimizza il tempo della richiesta tipica
    def both(title, key, flag, values, field, norm, to_param, from_param):
        if key in explicit:
            return
        log(f"\n== {title}")
        pp = best_of(b.measure("pp", cur["thread_prompt"], cur, flag, values), field, norm)
        tg = best_of(b.measure("tg", cur["thread"], cur, flag, values), field, norm)
        times = {k: b.request_time_se(pp[k], tg[k]) for k in pp if k in tg and pp[k][0] > 0 and tg[k][0] > 0}
        if not times:
            log("   nessuna misura valida, resta " + str(cur[key]))
            return
        for k in sorted(times, key=str):
            log(f"   {to_param(k)}: prompt {fmt_ts(pp[k])}, risposta {fmt_ts(tg[k])} token/s,"
                f" richiesta tipica {times[k][0]:.2f}±{times[k][1]:.2f} s")
        best, why = choose(times, from_param(cur[key]), False, lambda k: str(to_param(k)))
        cur[key] = to_param(best)
        log(f"   -> {why}")

    fa_to = {-1: "auto", 1: True, 0: False}
    fa_from = lambda v: 1 if v in (True, "on") else 0 if v in (False, "off") else -1
    if str(params["cache_v"]) == "f16":
        both("4. flash attention", "flash_attn", "-fa", ["auto", "on", "off"], "flash_attn", int, fa_to.get, fa_from)
    both("5. poll (attesa attiva dei thread)", "poll", "--poll", [int(x) for x in tcfg.get("poll", [0, 50])],
         "poll", int, int, int)
    if not ctx["big"]:
        both("6. repack dei pesi", "repack", "--repack", [0, 1], "repack", bool, bool, bool)

    # 7) thread fissati sui core P (solo se si usa un thread per core P)
    if "solo_core_p" not in explicit and ctx["maschera_p"] and n_p and cur["thread"] == n_p and cur["thread_prompt"] == n_p:
        log("\n== 7. thread fissati sui core P")
        t = {}
        for pin in (False, True):
            pp = best_of(b.measure("pp", n_p, dict(cur, solo_core_p=pin)), "n_threads")
            tg = best_of(b.measure("tg", n_p, dict(cur, solo_core_p=pin)), "n_threads")
            if pp and tg:
                t[pin] = b.request_time_se(pp[n_p], tg[n_p])
                log(f"   {'fissati' if pin else 'liberi'}: prompt {fmt_ts(pp[n_p])}, risposta {fmt_ts(tg[n_p])} token/s,"
                    f" richiesta tipica {t[pin][0]:.2f}±{t[pin][1]:.2f} s")
        if len(t) == 2:
            cur["solo_core_p"], why = choose(t, False, False, lambda k: "fissati" if k else "liberi")
            log(f"   -> {why}")

    # 8) decodifica speculativa: llama-bench non la supporta, si misura con llama-server su un testo vero
    if not ({"spec_tipo", "spec_n"} & explicit):
        cands = [("none", cur["spec_n"])]
        if ctx["draft"]:
            cands += [("draft-simple", int(n)) for n in tcfg.get("spec_token", [2, 4, 8])]
        if tcfg.get("spec_ngram"):
            cands.append(("ngram-simple", cur["spec_n"]))
        prompt_file = tcfg.get("prompt")
        prompt_path = os.path.join(ctx["base"], str(prompt_file)) if prompt_file not in (None, "none") else None
        if len(cands) > 1 and prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, encoding="utf-8-sig") as f:
                prompt = f.read()
            n_predict = 48 if ctx["big"] else 128
            reps = 2
            log(f"\n== 8. decodifica speculativa (llama-server, {os.path.basename(prompt_path)}, {n_predict} token, {reps} richieste)")
            times = {}
            name = lambda k: k[0] if k[0] != "draft-simple" else f"bozza, {k[1]} token"
            for tipo, n in cands:
                # parametri completi (contesto, cache, ...) con i valori gia' scelti da tune
                p = dict(params)
                p.update(cur)
                p.update(spec_tipo=tipo, spec_n=n)
                r = server_measure(b, p, prompt, n_predict, reps, report)
                if not r:
                    log(f"   {name((tipo, n))}: misura fallita (vedi rapporto)")
                    continue
                pp, tg, drafted, accepted = r
                times[(tipo, n)] = b.request_time_se(stats_of(pp), stats_of(tg))
                acc = f", bozze accettate {accepted}/{drafted} ({100 * accepted / drafted:.0f}%)" if drafted else ""
                log(f"   {name((tipo, n))}: prompt {fmt_ts(stats_of(pp))}, risposta {fmt_ts(stats_of(tg))} token/s,"
                    f" richiesta tipica {times[(tipo, n)][0]:.2f}±{times[(tipo, n)][1]:.2f} s{acc}")
            if times:
                current = ("none", cur["spec_n"])
                best, why = choose(times, current, False, name)
                cur["spec_tipo"], cur["spec_n"] = best
                log(f"   -> {why}")
        elif len(cands) > 1:
            log("\n== 8. decodifica speculativa: saltata, manca il testo di prova (tune.prompt)")

    # misura finale con i valori scelti
    log("\n== misura finale")
    pp = best_of(b.measure("pp", cur["thread_prompt"], cur), "n_threads")
    tg = best_of(b.measure("tg", cur["thread"], cur), "n_threads")
    pp_v = next(iter(pp.values()), (0.0, 0.0, 0))[0]
    tg_v = next(iter(tg.values()), (0.0, 0.0, 0))[0]
    t_req = b.request_time(pp_v, tg_v) if pp_v and tg_v else None
    log(f"   prompt {pp_v:.1f} token/s, risposta {tg_v:.1f} token/s"
        + (f", richiesta tipica {t_req:.1f} s" if t_req else ""))
    if noise:
        med = sorted(noise)[len(noise) // 2]
        log(f"   incertezza tipica delle misure: {med:.1%}")
        if med > soglia:
            log("   ATTENZIONE: misure rumorose (il PC era occupato?). Alcune scelte sono rimaste ai valori prudenti;"
                " per una calibrazione migliore rilanciare 'stem tune' a PC tranquillo o aumentare 'giri'.")

    cfg = ctx["cfg"]
    out = {
        "modello": cfg.get("modello"),
        "modello_byte": os.path.getsize(ctx["model"]),
        "ram_gb": (cfg.get("memoria") or {}).get("ram_gb"),
        "contesto": (cfg.get("llama") or {}).get("contesto"),
        "data": t0.strftime("%Y-%m-%d %H:%M"),
        "valori": {k: cur[k] for k in cur if k not in explicit},
        "misure": {"prompt_token_s": round(pp_v, 2), "risposta_token_s": round(tg_v, 2),
                   "richiesta_tipica_s": round(t_req, 2) if t_req else None},
    }
    with open(tuned_path, "w", encoding="utf-8") as f:
        f.write('# generato da "stem tune": non modificare a mano, rilanciare "stem tune"\n')
        f.write(yaml_dump(out) + "\n")
    os.makedirs(RESULTS, exist_ok=True)
    rep = os.path.join(RESULTS, f"{t0:%Y-%m-%d_%H%M}-tune.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"\nstem: scelte salvate in {tuned_path}\n      rapporto completo in {rep}  ({dt / 60:.1f} min)")


# =========================================================================== show / run / serve

def cmd_show(args):
    cfg_path, tuned_path = config_paths(args)
    p, origin, ctx = resolve(cfg_path, tuned_path)
    f = ctx["facts"]
    print(f"modello  : {ctx['model']} ({f['size'] / 1e9:.2f} GB)")
    print(f"memoria  : servono circa {ctx['need'] / GIB:.1f} GiB, limite {ctx['limit'] / GIB:.1f} GiB"
          + (" -> modello GRANDE: mmap senza repack, tetto di memoria" if ctx["big"] else " -> sta in memoria"))
    bin_dir = ctx["engines"][p["motore"]]
    devs = list_devices(bin_dir)
    print(f"motore   : {p['motore']} ({bin_dir})")
    print("GPU      : " + ("; ".join(f"{d['nome']} {d['descrizione']} {d['mib']} MiB"
                                    f" ({'integrata' if d['integrata'] else 'dedicata' if d['integrata'] is False else '?'})"
                                    for d in devs) if devs else "nessuna per questo motore"))
    print("\nparametri:")
    keys = ["contesto", "thread", "thread_prompt", "batch", "ubatch", "flash_attn", "cache_k", "cache_v",
            "repack", "poll", "solo_core_p", "motore"]
    if devs:
        keys += ["dispositivi", "strati", "esperti_su_cpu", "fit", "op_offload"]
    keys += ["spec_tipo", "spec_n"]
    for k in keys:
        print(f"  {k:14s} {str(p[k]):10s} ({origin[k]})")
    print(f"  {'bozza':14s} {ctx['draft'] or 'nessuna'}")
    print(f"  {'parallelo':14s} {server_parallel(ctx)} (server)")
    if os.path.exists(tuned_path):
        m = (yaml_load(tuned_path).get("misure") or {})
        if m:
            print(f"\nultima misura di tune: prompt {m.get('prompt_token_s')} token/s,"
                  f" risposta {m.get('risposta_token_s')} token/s, richiesta tipica {m.get('richiesta_tipica_s')} s")
    print("\nchat   : " + " ".join(quote(x) for x in limiter(ctx) + [exe("llama-cli", bin_dir)] + llama_args(p, ctx)))
    print("server : " + " ".join(quote(x) for x in limiter(ctx) + [exe("llama-server", bin_dir)] + llama_args(p, ctx)
                                  + server_args(ctx)))


def server_parallel(ctx):
    return max(1, int(ctx["server"].get("parallelo", 1) or 1))


def server_args(ctx):
    srv = ctx["server"]
    a = ["--host", str(srv.get("host", "127.0.0.1")), "--port", str(srv.get("porta", 8080))]
    n = server_parallel(ctx)
    return a + (["-np", str(n)] if n > 1 else [])


def launch(args, tool, extra):
    cfg_path, tuned_path = config_paths(args)
    p, origin, ctx = resolve(cfg_path, tuned_path)
    if tool == "llama-server":
        extra = server_args(ctx) + extra
    cmd = limiter(ctx) + [exe(tool, ctx["engines"][p["motore"]])] + llama_args(p, ctx) + extra
    print("stem: " + " ".join(quote(x) for x in cmd), file=sys.stderr)
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def cmd_run(args):
    return launch(args, "llama-cli", args.extra)


def cmd_serve(args):
    return launch(args, "llama-server", args.extra)


# =========================================================================== main

def main():
    ap = argparse.ArgumentParser(prog="stem", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "stem.yaml"), help="file di configurazione (default: stem.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="crea stem.yaml")
    p.add_argument("-m", "--modello", required=True)
    p.add_argument("--template", choices=sorted(TEMPLATES), default="chat")
    p.add_argument("--ram-gb", type=float, default=4.0, help="RAM massima per llama.cpp in GB (0 = nessun limite)")
    p.add_argument("--forza", action="store_true", help="sovrascrive stem.yaml")
    p.add_argument("--llama", action="append", metavar="CARTELLA",
                   help="cartella con llama-bench, llama-cli e llama-server (ripetibile per piu' build)")
    p.set_defaults(fn=cmd_init)

    sub.add_parser("tune", help="misura e sceglie i parametri migliori").set_defaults(fn=cmd_tune)
    sub.add_parser("show", help="mostra parametri e comandi").set_defaults(fn=cmd_show)
    for name, fn, h in (("run", cmd_run, "avvia la chat (llama-cli)"), ("serve", cmd_serve, "avvia il server (llama-server)")):
        p = sub.add_parser(name, help=h)
        p.add_argument("extra", nargs=argparse.REMAINDER, help="argomenti extra per llama.cpp, dopo --")
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    if getattr(args, "extra", None) and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
