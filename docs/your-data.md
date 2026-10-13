# Your things, outside the checkout

Updates replace the harness's code. If you keep your own personas, skills, settings and data in the
same folder as that code, every update has to work around them. A home folder avoids that, the way
Hermes Agent keeps everything of yours in `~/.hermes/`: updates never touch a file in it.

The home folder is `~/.jevh` (or wherever `JEVH_HOME` points). It is used only if it exists.

```text
~/.jevh/
├── config.yaml     your settings, laid over the checkout's config.yaml (relative paths: relative to ~/.jevh)
├── personas/       your personas; one named like a built-in persona replaces it
├── skills/         your skills (same rule), learned skills (learned/) and lesson notes (_lessons/)
└── workspace/      memory, failure log, reviews, notebook, newsroom, usage, sessions, data snapshots
```

## Moving to it

```bash
jevh home          # where your things are now, and what moving would do (changes nothing)
jevh home move     # do it (asks first). Stop the CLI and dashboard first, and restart them after.
```

`jevh home move` doesn't delete anything. It moves things into the home folder, or copies them there
before it resets a checkout file to the shipped version:

| In the checkout | Goes to | Notes |
|---|---|---|
| `config.local.yaml` | `~/.jevh/config.yaml` | Relative paths in it become absolute, so they point at the same folders. The original is kept as `config.local.yaml.old`. |
| Settings you changed in `config.yaml` | `~/.jevh/config.yaml` | Only the keys you changed. `config.yaml` is reset to the shipped version. |
| `workspace/` | `~/.jevh/workspace/` | Your memory, logs and sessions. Moved only when it's inside the checkout. |
| `skills/learned/*`, `skills/_lessons/*` | `~/.jevh/skills/learned/`, `~/.jevh/skills/_lessons/` | Shipped lesson files stay. Lessons you added to one are moved out to a note. |
| Lessons the harness appended to a built-in `SKILL.md` | `~/.jevh/skills/_lessons/<name>.md` | The built-in skill goes back to the shipped text and keeps getting updates. The lessons are still added to it when it loads. |
| Personas and skills you added | `~/.jevh/personas/`, `~/.jevh/skills/` | |
| Built-in personas and skills you edited | a copy in `~/.jevh/` | Yours replaces the built-in one by name, and the checkout's file is reset. Your copy no longer gets updates (see below). |
| Edits to the code | stay | They're listed. Keep them on a git branch, or send them upstream. |

To check the result, run `jevh home` again; it should say nothing of yours is left in the checkout.
Then `git status` in the checkout is clean, apart from code you added yourself.

## How the harness reads it

Settings are applied in this order, and the last one wins (`jevh/config.py`):

1. `config.yaml` in the checkout. It ships with the harness and changes with updates.
2. `~/.jevh/config.yaml`. Relative paths in it are relative to `~/.jevh`.
3. `config.local.yaml` in the checkout, if you still have one. It is ignored by git.
4. `workspace/settings.yaml`: what the dashboard's **Settings** tab saved (the models and the web search engine, nothing else is read
   from it). It is your data, like the rest of `workspace/`, and it wins over every file above; delete it (or press Reset in the
   tab) to go back to them. See [the dashboard guide](dashboard.md#settings).

Personas are searched in `~/.jevh/personas` first, then the built-in `personas/`. Skills are searched
in `~/.jevh/skills` first, then every folder in `skills_dirs`. In both cases the first one found with
a given name wins. Learned skills and lesson notes are written to `~/.jevh/skills`. Lesson notes from
every skills folder are added to their skill, including the notes that ship with the harness.

A persona or skill that can't be used is skipped with the reason, instead of stopping the harness.
The reason could be bad YAML, a missing description, a list written as a single value, or a model
provider that isn't configured. The CLI lists skipped items when it starts, and so does the
dashboard (Settings, Status). An unknown key in a persona is dropped with a warning, and the persona still
loads. An MCP server that fails to start is listed the same way.

## Recommendations

- **Don't edit tracked files to customize the harness.** Put the change in `~/.jevh` instead.
  - To change a setting, add only that key to `~/.jevh/config.yaml`. New settings shipped later still
    reach you, because missing keys come from the checkout's `config.yaml`.
  - To change a persona, copy it to `~/.jevh/personas/` under the same name and edit the copy. To add
    one, create a new file there.
  - To change a skill, copy its folder to `~/.jevh/skills/` and edit the copy. For an extra skills
    root (such as a Hermes skills folder), add it to `skills_dirs` in `~/.jevh/config.yaml`.
- **Overriding has a cost.** Your copy of a built-in persona or skill no longer gets updates. Check now
  and then what changed upstream, for example with `git log -p personas/<name>.yaml`, and decide
  whether to take it. If you only wanted to add guidance to a skill, a note in
  `~/.jevh/skills/_lessons/<skill>.md` is better: it's added to the built-in skill, and the skill still
  gets updates.
- **Secrets** stay in environment variables (`TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`,
  `NANGO_SECRET_KEY`), not in any config file. `providers.<name>.api_key_env` names the variable.
- **Back it up.** `~/.jevh` is everything that's yours. Copy that one folder to back up or move an
  install. A fresh `git clone` of the harness plus `~/.jevh` gives you the same setup back.
- **Don't put the home folder in the checkout.** It must be outside it, or updates can see it as
  untracked files.
- **Run more than one setup** by pointing `JEVH_HOME` at different folders, for example a test
  setup next to your real one.

## Updates and your data

Before every update and undo, `jevh update` snapshots your data into `<workspace>/snapshots/` and keeps
the last 5 (`jevh/snapshot.py`). A snapshot holds the workspace without its session folders, plus
learned skills, lesson notes and config files. Undoing an update puts the old code back but leaves
your data as it is, because it holds what you did since. If the older code can't read data the newer
version changed, put the snapshot from before the update back:

```bash
jevh update --snapshots
jevh update --restore-data <name>     # with the CLI and dashboard stopped
```

## For agents changing this repository

When a user asks you to customize their harness (a persona, a skill, a setting), make the change in
their home folder as described above, not in tracked files. Run `jevh home` to see where things are.
If the home folder doesn't exist yet, say so and suggest `jevh home move` rather than editing tracked
files. Change tracked files only when the task is to change the harness itself, for everyone.
