# Setting up jev-harness on a Mac (hosted Jev + LM Studio)

This gets the harness running on macOS (Apple Silicon) with:
- **Jev** (every routing/gating decision, and the browser tool's clicks) on the hosted Typesafe API, so
  there is no local Jev model to run;
- **the workers** (the models that write the code and answers) in LM Studio on the Mac.

Linux works the same way.

## 1. What you need

- Python **3.11 or newer** (`python3 --version`; `brew install python@3.12` if it's older).
- **LM Studio** with at least one model downloaded. The harness talks to its local server.
- A **Typesafe API key** for the hosted Jev.
- About 32 GB of unified memory is comfortable for a ~27-35B worker; with 16 GB use a 7-14B model.

## 2. Get the code and install

```bash
git clone <repo-url> jev-harness
cd jev-harness
python3 -m venv .venv
source .venv/bin/activate          # do this in every new terminal before running the harness
pip install -e ".[web,browser,dev]"
python -m playwright install chromium
```

## 3. LM Studio

1. Download the models you want to use (a bigger one for hard tasks, optionally a smaller fast one).
2. Developer tab -> start the server (default `http://127.0.0.1:1234`). Leave "Just-in-Time model
   loading" on: the harness loads and unloads models itself, one at a time.
3. Note each model's id exactly as LM Studio shows it (e.g. `qwen3.8-27b`), or run `lms ls`.
4. The models need tool calling (most recent Qwen, Gemma and Llama models have it).

## 4. Your settings

```bash
cp config.local.example.yaml config.local.yaml
```

Edit `config.local.yaml` (git ignores it, so pulls never touch it):
- `models.strong` / `models.fast`: your LM Studio model ids. Delete `fast` to use one model for
  everything.
- Everything else in the example is already set for the hosted Jev.

Put your key in your shell profile (`~/.zshrc`), then open a new terminal:

```bash
export TYPESAFE_API_KEY="your-key-here"
```

## 5. Check it works

```bash
python -m pytest -q                               # offline tests (no models needed), ~2-3 minutes
python -m jevh.cli -c "what's 17 * 23? use code"   # one real turn: Jev + your LM Studio model
python -m jevh.web                                # the dashboard: http://127.0.0.1:8740/
```

Browser check: `python -m jevh.cli -c "open news.ycombinator.com in the browser and tell me the top story"`.
To watch the browser work, type `/watch` in the CLI chat (or tick "Watch browser" in the dashboard).

## Good to know

- **The Jev thresholds were tuned on the local Jev (semif).** The hosted Jev may score a little
  differently, so if routing or skill picks look off, run `python evals/routing_cases.py --sweep`
  and `python evals/skill_cases.py` and share the output.
- **Real-website tests:** `python evals/bench/browser_tasks.py` (about 20-40 minutes; nothing is bought or
  signed into). Results land in `evals/bench/runs/browser/`.
- **The code runs on your machine.** Workers execute Python in a separate process with no sandbox, and
  the browser acts on real websites. Watch what it does, and don't leave it running unattended.
- **Not needed on a Mac:** the `openjev\*.bat` scripts in README (they start Jed's local Jev on Windows)
  and ComfyUI (the `comfyui_manager` persona just reports that ComfyUI isn't running).
- **Memory and learned skills** are written to `workspace/` and `skills/learned/` on your machine only.

## Troubleshooting

| You see | Fix |
|---|---|
| `set the TYPESAFE_API_KEY environment variable` | `export` it (step 4) in the terminal you run from |
| `systemone 401` / `403` | the key is wrong or has no access to `jev-latest` |
| connection refused on `127.0.0.1:1234` | start LM Studio's server (step 3) |
| `model not found` from LM Studio | `models.strong` / `models.fast` must match the ids LM Studio shows |
| the browser tool fails to start | `python -m playwright install chromium` (inside the venv) |
| `python: not found` in a worker's shell command | activate the venv (`source .venv/bin/activate`) |
