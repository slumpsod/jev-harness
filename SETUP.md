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
pip install -e ".[web,browser,desktop,dev]"   # (desktop: the tool that operates Windows apps; comtypes + pillow)
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
logins and refreshes the tokens. No token ever reaches a model, and anything that changes something
(sending, creating, deleting) asks you first.

### The easy way: ask the assistant

Say **"Connect my Google Calendar"** (or Gmail, GitHub, Slack, Notion...), or **"Set up Nango"**. The
assistant does the work and talks you through the rest, one step at a time (the `nango-setup` skill):

1. **Nango, the first time.** It starts Docker and Nango on this machine (about 400 MB; you need
   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed). If Docker Desktop on
   Windows is stuck on leftover files from its last shutdown, it asks before moving them aside. Nothing to
   configure: it keeps the key and address in `workspace/nango/`.
2. **At the app's website.** It reads you each step (for Google: make a project, turn on the API, the
   consent screen, publish it, create the OAuth client), opens that page in your own browser, and waits for
   your "Done". The redirect URL it gives you is exact.
3. **Its ID and secret.** It asks for the Client ID, then the secret; the secret is typed hidden, goes
   straight to Nango on this machine, and is not kept in the conversation, memory or logs.
4. **Sign in.** It opens Nango's sign-in page in your browser, waits until you've allowed access, and checks
   the connection works ("connected as you@gmail.com").

Want it to click through the website for you? Say so. It opens a window **you** sign in on (it never types
your password) and does the clicking, and it can read the Client ID and secret straight off the page so you
don't copy anything. Google may refuse to sign in to an automated window; if so it says so and carries on in
your own browser, because working around that would be wrong. The window stays signed in between tasks
(a profile in `workspace/nango/browser-profile`). **Untested on Google's own pages** (they need your sign-in):
this was only run live against a practice console, so the by-hand walk-through is the proven path.

Every time you add another app it's the same walk-through, and Google apps reuse the project. Apps that sign
in with a key or password (Todoist, Airtable, Apple Calendar...) need no app at all: it tells you where to
find the details and you type them on Nango's own page.

### Or by hand

Everything above is also plain Python the assistant calls: `nango.status()`, `nango.start()`,
`nango.guide(app)`, `nango.setup(app)`, `nango.connect_and_wait(app)`, `nango.test(app)`. You can call them
yourself in a terminal (`python -c "from jevh.tools import nango; nango.setup('google-mail', access='write')"`):
what they would ask you in the chat they ask in the terminal, a Client secret typed hidden. And the
long way, to see what it does:

**Nango on this machine** (Docker Desktop running):

```bash
cd nango
cp .env.example .env
openssl rand -base64 32       # put this after NANGO_ENCRYPTION_KEY= in .env, and keep a copy: it can't be replaced
docker compose up -d          # dashboard: http://localhost:3003
```

The dashboard asks you to sign up for an account (it stays in the local database); `FLAG_AUTH_ENABLED=false`
in `nango/.env` drops the login instead, leaving it open to anything running on this machine (the
assistant asks which you want, and can fetch the key itself only when there's no login). Everything listens
on 127.0.0.1 only; your connections live in the `nango-data` volume.

**The key.** Dashboard > Environment Settings > API Keys > the secret key. Make it available as
`NANGO_SECRET_KEY` in your environment (Windows: `setx NANGO_SECRET_KEY ...`, then a new terminal), or put
the key in `workspace/nango/secret_key` and `http://localhost:3003` in `workspace/nango/url`. Nango Cloud
instead of local: sign up at [app.nango.dev](https://app.nango.dev), use its secret key, and leave the URL out.

**An app at the provider.** For OAuth apps (Google, GitHub, Slack...), make an OAuth app at the provider
with the redirect URL `http://localhost:3003/oauth/callback`. For Google (Gmail, Calendar, Drive, Docs,
Sheets), once for all of them:
1. [Google Cloud console](https://console.cloud.google.com/projectcreate): make a project, and under
   **APIs & Services > Library** enable the APIs you want.
2. **Google Auth Platform** (the OAuth consent screen): External, your email as support and contact.
3. **Audience > Publish app.** In "Testing", Google's logins expire after 7 days. Publishing an app only
   you use normally needs no review (Google shows an "unverified app" warning you click past once); if
   Google asks for verification, check its current rules.
4. **Clients > Create client**: Web application, with the redirect URL above. Copy the Client ID and secret.
5. In Nango's dashboard, **Integrations > Set up** the app (key e.g. `google-mail`) with that ID, secret and
   scopes (e.g. `https://www.googleapis.com/auth/gmail.readonly`), then **Connections > Add** and sign in.

Check: `jevh -c "what's on my calendar tomorrow?"`.

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
| `Nango isn't set up yet` | ask the assistant to "Set up Nango" (or `nango.start()` in its REPL) |
| `no '<app>' account is connected` | ask the assistant to connect it ("Connect my Gmail") |
| `isn't set up in Nango` | ask the assistant to set that app up ("Connect my Slack") |
| `can't reach Nango at http://localhost:3003` | start Docker Desktop, then `docker compose up -d` in `nango/` |
| Docker Desktop (Windows): `remove ...\engine.sock` (or `dockerInference`): `The file cannot be accessed by the system` | a socket left by a crash that Windows won't delete: quit Docker, rename the folder it names (e.g. `%LOCALAPPDATA%\docker-secrets-engine` -> `docker-secrets-engine.old`), start Docker again |
