# stem - configuratore di llama.cpp guidato da benchmark

*English summary: `stem` picks the fastest llama.cpp settings for **your** machine and model by measuring them.
One editable YAML file (in the spirit of [Soup](https://github.com/SevGrigoryan/soup)), hardware detection
(hybrid P/E cores, RAM, any GPU backend seen by llama.cpp), benchmark-driven tuning with a statistical
decision rule, and a hard RAM cap so large models do not starve other programs. Documentation is in Italian.*

`stem` sceglie i parametri di [llama.cpp](https://github.com/ggml-org/llama.cpp) più veloci **per il tuo PC e
per il tuo modello**, misurandoli invece di indovinarli. Tutto sta in un file YAML modificabile a mano.

```text
stem init  -m modello.gguf --llama C:\llama.cpp\bin --ram-gb 4   # rileva la macchina e crea stem.yaml
stem tune                                              # misura le alternative e sceglie le migliori
stem show                                              # parametri scelti, da dove vengono, comandi completi
stem run                                               # chat (llama-cli)
stem serve                                             # server OpenAI-compatibile (llama-server)
```

## Cosa fa

- **`stem init`:**
  - rileva la CPU, compresi i core P ed E delle CPU ibride Intel, e la RAM;
  - chiede a llama.cpp quali GPU vede (`--list-devices`), quindi funziona con qualsiasi backend: CUDA, Vulkan,
    SYCL, HIP, Metal;
  - analizza il file GGUF: se è un MoE, quanti esperti ha, quanta memoria serve;
  - cerca da solo un modello "bozza" compatibile per la decodifica speculativa;
  - crea `stem.yaml` con un commento per ogni parametro.
- **`stem tune`** misura con `llama-bench` (e con `llama-server` dove serve):
  0. il motore (più build di llama.cpp possono essere messe in gara) e l'uso della GPU: solo CPU, tutto sulla
     GPU, metà dei layer, GPU tranne gli esperti MoE (`-ncmoe`), fit automatico (`-fit`), GPU solo per il
     prompt;
  1. thread per la generazione;
  2. thread per il prompt;
  3. `ubatch`;
  4. flash attention;
  5. `poll`;
  6. repack dei pesi;
  7. thread fissati sui core P (`-C <maschera> --cpu-strict 1`);
  8. decodifica speculativa con un modello bozza (`--spec-type draft-simple`), misurata con `llama-server` su
     un testo vero.

  La scelta minimizza il tempo di una **richiesta tipica** (per esempio 500 token di prompt e 300 di risposta,
  dal template). Le misure si ripetono in più giri a ordine alternato, e **un valore cambia solo se è più
  veloce almeno della soglia (3%) e almeno 2 volte l'incertezza della misura**: il rumore del PC non produce
  scelte casuali. Il rapporto completo va in `risultati/`.
- **`stem run` / `stem serve`** avviano llama.cpp con i parametri scelti. Se `ram_gb` è maggiore di 0, lo fanno
  sotto un tetto rigido di memoria residente (`run-limited`), così un modello grande non toglie RAM agli
  altri programmi.
- **Modelli più grandi della RAM:** `stem` se ne accorge e passa da solo a mmap senza repack (`-nr`) e a misure
  ridotte. Senza `-nr`, il backend CPU copia gran parte dei pesi Q4_K in memoria privata, che finisce nel file
  di paging.

## Requisiti

- Windows 10/11 x64. Il rilevamento dei core P/E e `run-limited` sono specifici di Windows; su altri sistemi
  `stem` non è stato provato.
- Python 3.9 o successivo, senza pacchetti aggiuntivi. Se PyYAML è installato viene usato; altrimenti c'è un
  lettore YAML interno.
- I programmi di llama.cpp: `llama-bench`, `llama-cli`, `llama-server`.

## Installazione

1. **llama.cpp.** Compilarlo seguendo [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
   oppure scaricare una release ufficiale.
   - La cartella con i programmi si indica a `stem init --llama <cartella>`. L'opzione si può ripetere per
     più build (per esempio CPU e CUDA): `stem tune` le mette in gara.
   - Senza `--llama`, `stem` cerca in `llama.cpp/build/bin` e in `llama-vulkan/` dentro questa cartella.
   - Dopo `init` le cartelle si possono cambiare in `stem.yaml`, sezione `motori`.
2. **`run-limited`** (facoltativo, serve per il tetto di RAM `ram_gb`). Da un "Developer PowerShell for VS",
   nella cartella di `stem`:
   ```text
   cl /nologo /O2 /EHsc /std:c++17 /DUNICODE /D_UNICODE tools\run-limited\run_limited.cpp /Fe:tools\run-limited\run-limited.exe /link psapi.lib
   ```
   Senza `run-limited` impostare `memoria.ram_gb: 0` (nessun tetto).
3. **I modelli GGUF** vanno dove si preferisce. Il percorso si passa a `stem init -m`.

## Il file `stem.yaml`

| sezione | contenuto |
| --- | --- |
| `modello`, `template` | modello GGUF e uso (`chat`, `codice`, `documenti`, `veloce`) |
| `richiesta_tipica` | token di prompt e di risposta da ottimizzare |
| `memoria` | `ram_gb` (tetto rigido, 0 = nessuno) e priorità del processo |
| `llama` | contesto, thread, batch, ubatch, flash attention, tipo della KV cache, repack, poll, core P |
| `motori`, `motore` | cartelle di llama.cpp disponibili e quale usare |
| `gpu` | dispositivi, layer sulla GPU, esperti MoE in RAM, fit automatico, op offload |
| `speculativa` | tipo, modello bozza, token proposti |
| `server` | host, porta, conversazioni parallele (`-np`) |
| `tune` | ripetizioni, giri, soglia, alternative da provare, testo di prova |
| `macchina` | rilevata da `init` (informativa) |

- **`auto`:** lascia scegliere a `stem tune`. Senza tune vale una regola prudente, per esempio un thread per
  core P, nessuna GPU integrata, nessuna decodifica speculativa.
- **Valori espliciti:** un valore scritto a mano ha sempre la precedenza e `tune` non lo prova.
- **Risultati di `tune`:** vanno in `<config>-ottimizzato.yaml`, da non modificare a mano. Vengono ignorati da
  soli se cambiano il modello, la RAM o il contesto.
- **Più configurazioni:** `stem --config altro.yaml init ...`.

## Struttura

| percorso | contenuto |
| --- | --- |
| `stem.cmd` | avvio su Windows (`stem <comando>`) |
| `tools/stem/stem.py` | il configuratore |
| `tools/gguf-moe-info.py` | analisi dei file GGUF, anche da sola: `python tools/gguf-moe-info.py modello.gguf` |
| `tools/run-limited/run_limited.cpp` | lanciatore con tetto rigido alla memoria residente e priorità bassa |
| `tools/moe-trace/prompt-*.txt` | testi usati per le misure con `llama-server` |
