# The local Jev

jev-harness asks every decision (which assistant, which skills, memory, tools, the final() checks...) of **Jev**, a
System One decision model: typed yes/no, choice and score answers with probabilities. Run it on your own GPU with the
two small servers in this folder, or use the hosted Jev instead (SETUP.md).

```
jev-harness ──POST /v1/systemone──► systemone.py (:33001) ──► semif_bridge.py (:33003) ──► Qwen3.5-4B on your GPU
              config.yaml jev.url    the adapter               the model (SemIf direct readout)
```

- **`semif_bridge.py`** loads [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) (TheoLeeCJ, MIT): Qwen3.5-4B
  (pinned revision) reading the probability of each option straight from its next-token logits, without generating
  text. About 9 GB of GPU memory in bfloat16 (measured on an RTX 4080: 8.7 GB peak with long inputs), ~0.2-0.6 s a
  decision once warm, ~30 s to load.
- **`systemone.py`** is the address the harness uses. It forwards to the bridge and drops a request whose caller hung up.
  Anything else that speaks System One can sit behind it (set `SEMIF_URL`).

These are the defaults `config.yaml` expects (`jev.url: http://127.0.0.1:33001`, `jev.backend: semif`,
`jev.backend_health: http://127.0.0.1:33003/health`): nothing to configure if you keep the ports.

## Install

A CUDA GPU with ~10 GB free (an Apple Silicon or CPU setup can run SemIf too, slowly: see SemIf-OpenJev's README).
Python 3.10-3.12.

```bash
cd openjev
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128   # the torch for your GPU/CUDA
pip install -r requirements.txt
```

The model downloads from Hugging Face on the first start (~9 GB; set `HF_HOME` to put it on a big drive).

## Run

Windows (two windows, or put both in your Startup folder):

```bat
set CUDA_VISIBLE_DEVICES=0
openjev\start_semif.bat
openjev\start_systemone.bat
```

Linux/macOS: `CUDA_VISIBLE_DEVICES=0 openjev/start.sh` starts both in the background (logs in `openjev/logs`).

Check: `curl http://127.0.0.1:33003/health` says `"status": "up"`, then `python systemone.py --health`, then the
dashboard's Status section (Settings → Status) shows the adapter and the backend up.

## Settings (environment)

| | default | |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | all | which GPU the bridge uses (keep it off the one your worker model runs on) |
| `SEMIF_PORT`, `SEMIF_HOST` | 33003, 127.0.0.1 | the bridge's address |
| `SEMIF_PREFIX_CACHE` | 1 | run a request's shared state once for all its questions (~38% faster; 0 turns it off) |
| `SEMIF_MAX_TOKENS` | 12288 | the longest input; measured safe on a 16 GB card |
| `SEMIF_SRC` | | a SemIf-OpenJev checkout's `src/`, instead of the pip install |
| `SYSTEMONE_PORT`, `SYSTEMONE_HOST` | 33001, 127.0.0.1 | the adapter's address |
| `SEMIF_URL` | http://127.0.0.1:33003 | where the adapter forwards |
| `SEMIF_TIMEOUT` | 300 | seconds a request may take |

Both servers listen on 127.0.0.1 only. They have no authentication: don't expose them to a network.

## Numbers

With this bridge the harness's live evals report routing 97% top-1 on `evals/routing_cases.py`, and the other evals in
the README's list. They were run against this code on an RTX 4080 (Windows 11); an independent rebuild of the same
idea (issue #6) reproduced the routing numbers.
