# Patch per llama.cpp: niente precaricamento dei modelli più grandi della RAM

*English summary: when a GGUF model is loaded with mmap, llama.cpp asks the OS to prefetch the whole file.
If the model is larger than free memory, that prefetch cannot succeed (the end of the file evicts the beginning):
it only wastes disk reads and, on Windows, drains available RAM so other programs lose resident memory. This
19-line patch skips the prefetch in that case. Measured on Windows with Qwen3-30B-A3B (18.6 GB, 16 GB of RAM,
`--no-repack`): minimum available RAM during load 2.5-2.9 GiB instead of 0.4-1.1 GiB, 4.5 GB less read from
disk per start, same generation speed. Documentation is in Italian.*

## Il problema

Quando carica un modello con mmap (il modo predefinito), llama.cpp chiede al sistema operativo di precaricare
**tutto** il file:
- su Windows con `PrefetchVirtualMemory`;
- su Linux e macOS con `posix_madvise(POSIX_MADV_WILLNEED)`.

Succede in `llama_model_loader::init_mappings` (`src/llama-model-loader.cpp`) e in `src/llama-mmap.cpp`.

Se il modello è più grande della RAM libera, il precaricamento non può riuscire: la fine del file sfratta
l'inizio, che andrà comunque riletto quando serve. Il risultato è solo lavoro in più:
- **letture inutili dal disco** a ogni avvio;
- su Windows, **la RAM disponibile scende quasi a zero** durante il caricamento e il sistema toglie memoria
  residente agli altri programmi aperti.

## Cosa fa la patch

Aggiunge 19 righe all'inizio di `llama_model_loader::init_mappings`:
1. somma le dimensioni dei file del modello;
2. chiede la memoria libera al dispositivo CPU di ggml (`ggml_backend_dev_memory`);
3. se il modello è più grande, disattiva il precaricamento e lo scrive nel log:

```text
init_mappings: model size (17697 MiB) exceeds free memory (7945 MiB), skipping prefetch
```

Tutto il resto resta com'è. Il modello è sempre mappato con mmap e le pagine si leggono dal disco quando
servono. Con un modello che sta nella RAM libera non cambia niente.

**Limiti:**
- **Interviene solo** con mmap e con il precaricamento attivo, cioè nel caso predefinito.
- **La "memoria libera" viene dal backend CPU di ggml:**
  - su Windows è la RAM disponibile (`GlobalMemoryStatusEx`, `ullAvailPhys`);
  - su Linux e macOS ggml considera libera tutta la RAM, quindi la patch interviene solo con un modello più
    grande della RAM totale.
- **Provata solo su Windows 11.**
- **Versione di llama.cpp:** scritta e verificata sul commit `43fe9c6` (6 ottobre 2026). Su versioni
  successive può servire adattarla a mano: le righe da aggiungere sono poche.

## Misure

- **Macchina:** Windows 11, 16 GB di RAM, NVMe, altri programmi aperti (4.3-4.7 GiB disponibili all'avvio).
- **Modello:** Qwen3-30B-A3B Q4_K_M, 18.6 GB, più grande della RAM.
- **Confronto:** la stessa build di llama.cpp, compilata due volte, con e senza la patch.
- **Prova:** prompt di 8 token, 16 token generati, temperatura 0, 6 thread, contesto 512, `--no-repack`.
  La memoria residente del processo è limitata a 1.5 GB, per non togliere RAM agli altri programmi.
- **Ordine:** 6 esecuzioni alternate, 3 per variante.

| misura | senza patch | con patch |
| --- | ---: | ---: |
| tempo totale | 23.5-23.8 s | 21.4-22.2 s |
| prompt + generazione | 20.8-21.0 s | 20.8-21.5 s |
| letti dal disco (tutto il sistema) | 12.7-12.9 GB | **8.2-8.5 GB** |
| RAM disponibile minima | 0.4-1.1 GiB | **2.5-2.9 GiB** |
| memoria residente degli altri programmi | -40, -303, -372 MB | +90, -46, +388 MB |

**In sintesi:**
- **Gli altri programmi:** con la patch la RAM disponibile resta sopra i 2.5 GiB e non perdono memoria in
  modo sistematico.
- **Il disco:** si leggono circa 4.5 GB in meno a ogni avvio, e il caricamento è circa 1.7 s più veloce.
- **La velocità:** prompt e generazione vanno come prima.

### Prima di tutto `--no-repack`

Con i modelli più grandi della RAM, il problema più grande non è il precaricamento ma il **repack**.
- **Cosa fa il repack:** con AVX2 il backend CPU riorganizza i pesi Q4_K, esperti MoE compresi, in un buffer
  privato (`CPU_REPACK`).
- **Cosa succede con un modello più grande della RAM:** quella copia finisce nel file di paging. Su Qwen3, una
  prova breve (caricamento, 8 token di prompt e 16 generati) durava 149 s con il repack e 14 s senza.
- **Cosa fare:** usare `-nr` (`--no-repack`), che lascia tutti i pesi su mmap. La patch da sola non risolve
  questo problema; insieme a `-nr` toglie il precaricamento inutile.

## Come applicarla

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 43fe9c6
git apply ../0001-skip-prefetch-when-model-exceeds-free-memory.patch
```

1. **Compilare llama.cpp come al solito:**
   [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
2. **Verificare:** caricando un modello più grande della RAM libera, il log mostra `skipping prefetch`.
3. **Per toglierla:** `git apply --reverse ../0001-skip-prefetch-when-model-exceeds-free-memory.patch`.

## Contenuto

| file | contenuto |
| --- | --- |
| `0001-skip-prefetch-when-model-exceeds-free-memory.patch` | la patch (formato `git diff`) |
| `README.md` | questa spiegazione |
