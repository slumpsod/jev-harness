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

## Smaller and faster (UNVERIFIED in daily use; measured 2026-10-21; none of these is the default)

**UNVERIFIED:** the numbers below come from the Jev evals and a replay of their requests, not from real sessions. Laya
and the 4-bit model lose cases on the strict gates (failure check, learning gate) at the thresholds tuned for semif, and
nobody has run the harness on them day to day. Treat them as experiments; semif stays the default.

Measured against semif on the same requests: every request the Jev evals send, split in half by the user's request, with
nothing from the scored half used for training or tuning (`evals/bench/system_one_swap.py`; the research scripts that
made the Laya checkpoint live outside the repo). "Agreement" is the share of yes/no answers on the same side of 0.5 and of
choices with the same leader; latency is one request at a time on an RTX 3090.

| | GPU memory | latency p50 | agreement yes/no / choice | harness evals at the shipped thresholds |
|---|---|---|---|---|
| semif (Qwen3.5-4B, bf16) | 8.7 GB peak | 0.22 s | - | routing 97.3%, failure check 7/7, learning gate 11/12 |
| semif + flash-linear-attention | 8.7 GB | **0.16 s** | 99.9% / 97% | same model: not re-run |
| semif 4-bit (`SEMIF_QUANT=4bit`) | **3.2 GB** | 0.39 s | 93% / 88% | routing 94.6%, failure check 4/7, learning gate 8/12 |
| Laya-large, fine-tuned (`laya_bridge.py`) | **~2 GB, or CPU** | **0.03 s** (CPU 0.65 s) | 89% / 77% | routing 91.9%, failure check 3/7, learning gate 4/12 |
| **Laya-multilingual v2** (the CPU checkpoint, 2026-10-22) | ~2 GB, or CPU (~3.4 GB RAM) | 0.035 s (CPU 0.24 s) | 88% / 73% | routing 89.2%, collaborators 91.9% (semif 86.5%), memory recall 84%; strict gates handed off in light mode |
| SemIf on Qwen3.5-2B / 0.8B | 3.5 / 1.4 GB | 0.11 / 0.17 s | 71% / 63%, 45% / 47% | not run |

- **Faster, same answers:** install the linear-attention kernels into the bridge's environment (Windows:
  `pip install "triton-windows<3.5" flash-linear-attention`; measured with torch 2.8, not yet with the torch 2.10 above).
  The "falling back to its reference PyTorch implementation" line in the bridge's log goes away.
- **Less memory:** the 4-bit model and Laya both lose cases on the strict gates (the failure check and the learning
  gate fire at 0.85-0.9, where their confidence differs from semif's). Use them where memory matters more than those.

### Light Jev (EXPERIMENTAL, UNVERIFIED)

For a computer without a spare GPU: Settings > Decisions (Jev) > Light, or `jev.mode: light` in config.yaml. Laya, on
the CPU, makes the decisions it was measured to make well (routing, skills, tools, memory, tier, collaborators,
interjection); the answer, failure, progress, code-risk, follow-up and learning checks are handed off: to the task's
own model as one quick question each ("Ask the task's model", the default) or skipped ("Off", as when Jev is down).
`jevh/light.py` says which decision goes where.

One short live run so far (`evals/bench/light_jev.py`, 6 tasks x 3 arms, 2026-10-22, Qwen3.8-27B worker, Laya v2 on the
CPU, one repeat): full semif 5/6, light with the task's model 4/6, light with the checks off 3/6. The kept decisions
matched semif; the misses were handed-off checks (a missing file the task's model did not log as a failure; with the
checks off, a lasting fact not learned and a task left to run to the time limit). Turns were slower on that worker:
mean 53 s full, 180 s light (the task's model), 151 s light (off). A smoke test, not a measurement.

### Laya (`laya_bridge.py`, UNVERIFIED)

[Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0, ModernBERT-large, 421M parameters) trained to give
semif's probabilities on harness decisions. Out of the box it agrees with semif on ~72% of yes/no answers and ~45% of
choices; after training, 89% and 77%. It is strong on the frequent, simple decisions (skills, memory, tools, devices,
collaborators: 90-97% agreement) and weak on the reasoning-heavy checks (rules, `project.find`, the failure and answer
checks: 40-80%).

```
jev-harness ──► systemone.py (:33001) ──"model": "laya"──► laya_bridge.py (:33004) ──► the Laya checkpoint (GPU or CPU)
                                       └─"model": "semif"─► semif_bridge.py (:33003)
```

Use it: put the checkpoint folder in `openjev/models/laya-harness` (or set `LAYA_CHECKPOINT`), `pip install laya==0.4.13`,
start `start_laya.bat` (CPU: `set LAYA_DEVICE=cpu` first) and the adapter as usual, and in `config.local.yaml`:

```yaml
jev:
  backend: laya
  backend_health: http://127.0.0.1:33004/health
```

Going back is deleting those lines; the adapter serves both. How the checkpoint was made: semif's answers to every eval
request and to Jed's own saved turns (put through the harness's Jev code paths); half of them seeded ~3,100 look-alike
cases written by a local LLM (Gemma 4 12B), which semif answered; Laya typed-decisions was trained on semif's
probabilities for ~4,850 requests (soft cross-entropy, 3 epochs, 2,048 tokens), and its temperatures fitted to semif's
on a slice it never trained on.

| `LAYA_*` setting | default | |
|---|---|---|
| `LAYA_CHECKPOINT` | openjev/models/laya-harness | the checkpoint folder |
| `LAYA_DEVICE` | auto | cuda or cpu |
| `LAYA_ONNX` | | an ONNX export of the checkpoint (a path, or a file name in the checkpoint folder): runs on the CPU through ONNX Runtime (`pip install onnxruntime`). Run `python laya_onnx.py FILE` on a fresh export once (needs `pip install onnx`): it fixes a NaN that Laya's ONNX runtime hits on batched questions. See "On the CPU" below before using it |
| `LAYA_ONNX_ARENA` | 1 | 0: ONNX Runtime frees its working memory after each request (~1.7 GB RAM instead of ~2.9 GB, about 4x slower) |
| `LAYA_THREADS` | 8 | CPU threads |
| `LAYA_BRIDGE_PORT`, `LAYA_BRIDGE_HOST` | 33004, 127.0.0.1 | the bridge's address |
| `LAYA_URL` (adapter) | http://127.0.0.1:33004 | where the adapter sends `"model": "laya"` |

#### On the CPU (measured 2026-10-22, Laya v2, i7-14700K, 8 threads)

PyTorch (the default) is the one to use. Through the bridge, 100 held-out harness requests one at a time:

| runtime | per request (p50) | RAM after 100 requests | agreement with semif, yes/no / choice | harness evals |
|---|---|---|---|---|
| PyTorch (`LAYA_DEVICE=cpu`) | 0.24 s | 3.4 GB | 87.5% / 69.4% | as the GPU run (same answers) |
| ONNX fp32 (`LAYA_ONNX`, fixed) | 0.25 s | 3.2 GB | 87.5% / 69.4% (same answers as PyTorch, within 1e-4) | same answers |
| ONNX int8 (`LAYA_ONNX`, fixed) | 0.14 s | 2.9 GB | 85.1% / 66.1% | worse: collaborators 83.8% (91.9%), memory recall 74% (84%), desktop "done" 6/12 (12/12), devices recall 80% (93%), Newsroom 74% (87%) |

- ONNX fp32 gives PyTorch's answers but is no faster through the bridge (0.19 s against 0.23 s in-process); int8 is
  ~2x faster and loses real decisions (Laya's own notes say accurate int8 needs quantization-aware training).
- An ONNX export straight from Laya's `export_onnx.py` returns NaN for about 1 in 7 requests with several questions of
  different lengths (agreement 85.6% instead of 87.5%): run `laya_onnx.py` on it. The bridge runs an unfixed file
  correctly but about twice as slowly, and says so in its log.
- On a CPU with performance and efficiency cores (Intel 12th-14th gen), Windows sometimes runs the bridge on the
  efficiency cores: the same requests then take 2-3x as long. Measured with the process kept on the performance cores.
- RAM is ~2 GB at start (Python, PyTorch, the model) and grows to the figure above as the largest requests come in.
