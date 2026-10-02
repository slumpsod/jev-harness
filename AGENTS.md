# Notes for coding agents

- **Customizing someone's setup is not changing the harness.** A user's personas, skills and settings
  belong in their home folder (`~/.jevh`, or `JEVH_HOME`), not in tracked files. Put a setting in
  `~/.jevh/config.yaml` (only the keys that change), a persona in `~/.jevh/personas/`, and a skill in
  `~/.jevh/skills/`. A file there with a built-in persona's or skill's name replaces it. Run
  `jevh home` to see what is where. See [docs/your-data.md](docs/your-data.md).
- Edit tracked files (`config.yaml`, `personas/`, `skills/`, `jevh/`) only to change the harness for
  everyone. Updates (`jevh update`) carry local edits along, but an edit can clash with an update and
  block it.
- Run the tests with `python -m pytest -q`. They run offline with a fake Jev and scripted models.
  A push to `main` becomes `stable`, the default update channel, only after
  `.github/workflows/stable.yml` passes.
- The architecture, with every loop and the data passed between them, is in
  [docs/architecture](docs/architecture/README.md).
