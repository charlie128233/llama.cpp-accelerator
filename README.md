# stem e una patch per llama.cpp

*English summary: two independent tools for running [llama.cpp](https://github.com/ggml-org/llama.cpp) well on
an ordinary PC, including Mixture-of-Experts models larger than RAM. **stem** is a benchmark-driven configurator
that picks the fastest llama.cpp settings for your machine and model from one editable YAML file. **The patch**
(19 lines) stops llama.cpp from prefetching a model that does not fit in free memory, which on Windows drains
available RAM and wastes disk reads. Each works without the other. Documentation is in Italian.*

Due strumenti indipendenti per far girare llama.cpp al meglio su un PC normale, anche con modelli MoE (Mixture
of Experts) più grandi della RAM.

| | cos'è | cartella |
| --- | --- | --- |
| **stem** | configuratore: misura e sceglie i parametri di llama.cpp più veloci per il PC e il modello | [`stem-configuratore/`](stem-configuratore/) |
| **patch** | modifica di 19 righe a llama.cpp: niente precaricamento dei modelli più grandi della RAM libera | [`llama-cpp-patch-prefetch/`](llama-cpp-patch-prefetch/) |

Ognuno funziona anche da solo: stem va con qualsiasi llama.cpp, e la patch non richiede stem.

## stem: il configuratore

llama.cpp ha decine di opzioni: thread, batch, flash attention, repack dei pesi, uso della GPU, esperti MoE in
RAM, decodifica speculativa e altre. Il valore migliore dipende dalla macchina e dal modello. stem lo trova
**misurandolo**, invece di indovinarlo.

```text
stem init  -m modello.gguf --llama <cartella di llama.cpp> --ram-gb 4   # rileva la macchina, crea stem.yaml
stem tune                                                               # misura le alternative e sceglie
stem show                                                               # parametri scelti e comandi completi
stem run                                                                # chat (llama-cli)
stem serve                                                              # server OpenAI-compatibile (llama-server)
```

- **Un file YAML** (`stem.yaml`) contiene tutti i parametri, commentati. `auto` lascia scegliere a `stem tune`;
  un valore scritto a mano ha sempre la precedenza.
- **`stem init` rileva la macchina:**
  - la CPU, con i core P ed E delle CPU ibride Intel, e la RAM;
  - le GPU viste da llama.cpp, con qualsiasi backend (CUDA, Vulkan, SYCL, HIP, Metal);
  - la struttura del modello GGUF;
  - un eventuale modello "bozza" per la decodifica speculativa.
- **`stem tune` misura**, con `llama-bench` e `llama-server`:
  - le build di llama.cpp e l'uso della GPU;
  - i thread, `ubatch`, flash attention, poll e repack;
  - i thread fissati sui core P;
  - la decodifica speculativa.

  Ottimizza il tempo di una **richiesta tipica**, per esempio 500 token di prompt e 300 di risposta.
- **La scelta è statistica:** le misure si ripetono in più giri, a ordine alternato. Un valore cambia solo se è
  più veloce almeno del 3% e di 2 volte l'incertezza della misura, così il rumore del PC non produce scelte a
  caso.
- **Il tetto di RAM** (`ram_gb`): `stem run` e `stem serve` avviano llama.cpp con un limite rigido alla memoria
  residente (`run-limited`, solo Windows). Così un modello grande non toglie RAM agli altri programmi.
- **I modelli più grandi della RAM** vengono riconosciuti: stem passa da solo a mmap senza repack (`-nr`) e a
  misure ridotte.

**Requisiti:** Windows 10/11 x64, Python 3.9 o successivo (nessun pacchetto aggiuntivo), i programmi di llama.cpp
(`llama-bench`, `llama-cli`, `llama-server`). Installazione e dettagli:
[`stem-configuratore/README.md`](stem-configuratore/README.md).

## La patch: niente precaricamento dei modelli più grandi della RAM

**Il problema.** Quando carica un modello con mmap (il modo predefinito), llama.cpp chiede al sistema operativo
di precaricare **tutto** il file. Se il modello è più grande della RAM libera, il precaricamento non può
riuscire, perché la fine del file sfratta l'inizio:
- legge dal disco gigabyte che poi vanno riletti;
- su Windows la RAM disponibile scende quasi a zero e gli altri programmi perdono memoria residente.

**La soluzione.** All'inizio di `llama_model_loader::init_mappings` la patch confronta la dimensione del modello
con la memoria libera. Se il modello è più grande, salta il precaricamento e lo scrive nel log
(`skipping prefetch`). Il resto non cambia: il modello resta in mmap e le pagine si leggono quando servono.

**Misure** su Windows 11, 16 GB di RAM, Qwen3-30B-A3B Q4_K_M (18.6 GB), `--no-repack`, altri programmi aperti.
Stessa build compilata con e senza patch, 3 esecuzioni alternate per variante:

| misura | senza patch | con patch |
| --- | ---: | ---: |
| RAM disponibile minima durante il caricamento | 0.4-1.1 GiB | **2.5-2.9 GiB** |
| letti dal disco a ogni avvio | 12.7-12.9 GB | **8.2-8.5 GB** |
| tempo totale | 23.5-23.8 s | 21.4-22.2 s |
| velocità di prompt e generazione | uguale | uguale |

**Limiti:**
- **Versione:** scritta e verificata su llama.cpp al commit `43fe9c6` (6 ottobre 2026).
- **Sistemi:** provata solo su Windows. Su Linux e macOS ggml considera libera tutta la RAM, quindi lì la patch
  interviene solo con modelli più grandi della RAM totale.

Spiegazione completa, tabella di tutte le misure e istruzioni per applicarla:
[`llama-cpp-patch-prefetch/README.md`](llama-cpp-patch-prefetch/README.md).

## Usarli insieme

Per un modello più grande della RAM, per esempio un MoE da 18 GB su un PC da 16 GB:
1. **Compilare llama.cpp con la patch:**
   ```bash
   git checkout 43fe9c6
   git apply 0001-skip-prefetch-when-model-exceeds-free-memory.patch
   ```
   Poi si compila come al solito.
2. **Configurarlo con stem:** `stem init -m modello.gguf --llama <cartella dei programmi> --ram-gb 4`, poi
   `stem tune`.
3. **Usarlo:** `stem run` o `stem serve`.

Ognuno copre una parte diversa del problema:

| chi | cosa fa | quando agisce |
| --- | --- | --- |
| stem | usa `-nr`, così i pesi restano in mmap e non finiscono nel file di paging | sempre, con i modelli più grandi della RAM |
| stem | limita la memoria residente di llama.cpp a `ram_gb` | dall'avvio alla fine |
| patch | evita il precaricamento inutile del file | durante il caricamento |

Il risultato: gli altri programmi aperti mantengono la loro memoria anche mentre il modello gira.
