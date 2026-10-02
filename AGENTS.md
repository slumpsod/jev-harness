# Notes for coding agents

## Work built in the cloud or remotely must ship with a local live test bench

**Read this before you add to the harness or change it.** The offline tests (`python -m pytest -q`) use a
fake Jev and scripted models. They prove the code runs, not that it works: the harness's behaviour comes
from real models (a Jev backend and worker LLMs) making judgement calls, and a fake never does. Cloud and
remote sessions cannot reach those models, so nothing built there has been seen to work until someone
runs it live. A change that only passes the offline tests is not finished.

If you build or change a feature in the cloud or on a remote machine, you must include a **test bench**
that Jed (or anyone with the local models) can run on their own machine against live models, and it must
be thorough:

- **Live models, no fakes.** It talks to the real Jev backend and real worker models through the
  harness's own code paths, the way `evals/*_cases.py` and `evals/bench/` do. Say which models it was
  designed for and what it needs (LM Studio, the hosted Jev, a browser, Nango).
- **Real cases, not toy ones.** Take inputs from real runs where there are any (`jevh failures export out.jsonl`,
  saved sessions). Include the cases that should fire, the ones that look similar and must not (false
  positives are how a gate breaks working tasks), and the hard or borderline ones. Add a **held-out
  split** (`--holdout`) so you can't tune to the cases you wrote, and a `--offline` re-sweep of saved raw
  scores, like the existing evals.
- **Measure the change, not just the feature.** Compare old against new on the same cases, and say what
  got better and what got worse. Include a regression set for what already worked, so a fix that breaks
  the harness shows up.
- **Repeat it.** Local models vary from run to run: run each case several times and report rates, not one
  pass. Report cost too (model calls, tokens, time) when the change adds work.
- **Run end to end as well.** Unit-style cases for each decision, plus full harness turns on real tasks
  when the feature touches the worker, the gate, hand-offs, tools or the dashboard.
- **Be runnable and readable.** One command, a plain-text summary with pass rates and the failing cases
  named, raw results saved beside it (`*_raw.json`), and a short note in the README's eval list saying
  what it checks and how to run it. Nothing it needs may be secret or machine-specific.
- **Say what you could not run.** If you could not run the bench yourself (you usually can't from the
  cloud), say so in the commit message and the PR, list the thresholds and numbers that are untuned
  guesses, and say what result would mean the feature is wrong. Never report numbers you did not
  measure.

Land the bench in the same change as the feature. Jed runs it locally before the feature counts as done;
until then, mark the feature as unverified in its docs.

## Working on the harness

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
