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

This gives you the `jevh` command while the venv is active. To use `jevh` from any terminal without
activating anything, also install it with pipx: see [the `jevh` command](README.md#the-jevh-command).

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
python -m pytest -q                               # offline tests (no models needed), about a minute or two
jevh -c "what's 17 * 23? use code"   # one real turn: Jev + your LM Studio model
jevh                                # chat in the terminal
jevh dashboard                      # the dashboard: http://127.0.0.1:8740/
```

Browser check: `jevh -c "open news.ycombinator.com in the browser and tell me the top story"`.
To watch the browser work, type `/watch` in the CLI chat (or tick "Watch browser" in the dashboard).

## Connected accounts (Nango)

The `nango` tool lets the assistants use your own accounts (email, calendar, Drive, GitHub, Slack, Notion,
accounting, fitness trackers... about 1,000 apps) through [Nango](https://nango.dev), which keeps the
logins and refreshes the tokens. It's optional: without a key the tool is simply not offered.

Run Nango on your own machine (below) or use [Nango Cloud](https://app.nango.dev) (skip to step 3 and
leave `nango_url` out). Local keeps your logins in a database on your machine, your email and files come
from the app straight to you, and the secret key only works from this machine. It costs a Docker
container and making your own OAuth app per provider (step 4).

### 1. Start Nango (Docker)

Needs [Docker Desktop](https://www.docker.com/products/docker-desktop/), running.

```bash
cd nango
cp .env.example .env
openssl rand -base64 32
```

Put that key after `NANGO_ENCRYPTION_KEY=` in `nango/.env`, and keep a copy somewhere safe: it can't be
changed later, and the stored logins can't be read without it. The dashboard asks you to sign up for an
account (it stays in the local database); `FLAG_AUTH_ENABLED=false` in `nango/.env` drops the login instead,
leaving the dashboard open to anything running on this machine. Then:

```bash
docker compose up -d
```

The dashboard is at http://localhost:3003. It restarts with Docker; `docker compose down` stops it (your
connections stay in the `nango-data` volume). Everything listens on 127.0.0.1 only.

### 2. Point the harness at it

In `config.local.yaml`:

```yaml
tools:
  nango_url: http://localhost:3003
```

### 3. The secret key

In the dashboard, open **Environment Settings > API Keys** and copy the secret key. Set it where you start the
harness: `export NANGO_SECRET_KEY=...` in your shell profile (Windows: `setx NANGO_SECRET_KEY ...`, then
open a new terminal).

### 4. Set up the apps you want

In the dashboard, **Integrations > Set up** each app (e.g. Gmail, Google Calendar, GitHub). The unique key you
give it (e.g. `google-mail`) is the name the assistants use. Apps that log in with an API key (Stripe,
Airtable PAT, Twilio, ...) need nothing more. OAuth apps need an OAuth app of your own at the provider, with
this redirect URL: `http://localhost:3003/oauth/callback`. Nango's page for each app links a guide.

Google (Gmail, Calendar, Drive, Docs, Sheets), once for all of them:
1. [Google Cloud console](https://console.cloud.google.com): make a project, and under **APIs & Services >
   Library** enable the APIs you want (Gmail API, Google Calendar API, Google Drive API...).
2. **OAuth consent screen**: External, your own email as a test user.
3. **Credentials > Create credentials > OAuth client ID**: Web application, with the redirect URL above.
4. Paste the client ID and secret into each Google integration in Nango, with its scopes, e.g.
   `https://www.googleapis.com/auth/gmail.modify` or `https://www.googleapis.com/auth/calendar`.
5. **The 7-day catch:** while the consent screen is in "Testing", Google's refresh tokens expire after
   7 days and you would reconnect weekly. Click **Publish app**: for your own account that's allowed
   without Google's review, and you click past an "unverified app" warning once when you connect.

### 5. Connect your accounts

On the dashboard's Connections page, or just ask ("connect my Gmail"): the assistant gives you a link, you
sign in, and reply "done".

Check: `jevh -c "what's on my calendar tomorrow?"`. When an assistant wants to send, create,
change or delete something, you're asked first and shown exactly what goes out.

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
| `Nango isn't set up: NANGO_SECRET_KEY is not set` | set it (Connected accounts, step 3) in the terminal you run from |
| `no '<app>' account is connected` | connect it in the Nango dashboard, or let the assistant give you a link |
| `isn't set up in Nango` | add that integration in the dashboard first (Connected accounts, step 4) |
| `can't reach Nango at http://localhost:3003` | start Docker Desktop, then `docker compose up -d` in `nango/` |
| Docker Desktop (Windows): `remove ...\engine.sock` (or `dockerInference`): `The file cannot be accessed by the system` | a socket left by a crash that Windows won't delete: quit Docker, rename the folder it names (e.g. `%LOCALAPPDATA%\docker-secrets-engine` -> `docker-secrets-engine.old`), start Docker again |
