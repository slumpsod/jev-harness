# 07 · The final() gate

When a worker calls `final()`, or replies with plain text and no code, `Worker._bounce`
(`worker.py:966`) runs the verifier before accepting. Problems go back to the worker as a new
message ("Not accepted yet: …"), with its REPL state intact. After `verify.max_bounces` (2), or
on a stalled worker's wrap-up step, the answer is accepted with the unresolved problems listed
under it. It is never accepted silently.

```mermaid
sequenceDiagram
  autonumber
  participant W as Worker._bounce
  participant VF as _verify_fn (orchestrator.py:661)
  participant V as Verifier.check
  participant FS as files / headless Chromium
  participant PT as project tests
  participant J as Jev
  participant RV as reviewer model

  W->>VF: answer, artifacts, messages (+ what final's block printed), changed, task_idx
  VF->>V: request, answer, artifacts, work_log(transcript), project, changed,<br/>evidence = seen_text(transcript), backups, review()
  loop each artifact
    V->>FS: exists, non-empty, image magic, .py compiles, .json parses,<br/>placeholders, missing local links
  end
  opt HTML artifacts and verify.render_html
    V->>FS: open in headless Chromium: script errors, blank page
  end
  opt worker changed project files
    V->>PT: verify.project_command or detected (npm test / pytest -q -x)
    opt tests pass and verify.review
      V->>J: risk per changed file (diff from .project_backups)
      opt any file ≥ review_risk 0.65
        V->>J: severity (or severe-looking diff regex) → strong / fast
        V->>RV: REVIEW_PROMPT with diffs → "NO PROBLEMS" or findings
      end
    end
  end
  alt answer is a bare JSON value
    Note over V: answer check skipped (data makes no claims)
  else verify.check_answer
    V->>J: verify: claims, delivers, honest, accurate
    V->>V: answer_rules(): grounding, counts, browser claims, promises
  end
  V-->>W: Verdict ok, problems, checks, scores
  W-->>W: event verify {ok, problems, checks, scores, bounced}
  alt problems and bounces left
    opt fast tier
      W->>W: escalate: swap in the strong model (event escalate)
    end
    W->>W: append "[Not accepted yet … Fix these and call final() again]"
  else
    W->>W: answer + "Not verified - the harness's checks still found: …" if any
  end
```

## Answer rules (`answer_rules`, `verify.py:194`)

| Problem raised | Condition |
|---|---|
| Unsupported numbers | the answer has 2+ digit numbers (not years) that appear nowhere in the request, prior turns, or tool outputs (`grounding`, `verify.py:167`); only once tools ran. Not counted against it (live false bounces, 2026-10-12): a rate a source prints without its zero (`$.49` is 0.49); a total the answer works out in the open from numbers the tools gave (`$2,800 ($2,000 + $800)`, `_derived`, sums and products, one level of totals of totals); a duration estimate (`a 30-second fix`); an approximation (`~20`, `about 20`) within 15% of a figure the tools gave. The replay in `evals/bench/flagged.py --stage replay` re-checks every draft your gate sent back over numbers |
| Claims not in the log (first of these three that applies) | `claims ≥ verify_claims 0.80` |
| … else doesn't deliver | `delivers < 0.80` and not `honest ≥ 0.80` |
| … else inaccurate report | code was run (the log isn't empty), `accurate < verify_accurate 0.40`, not honest, and the numbers aren't grounded (2+ supported, none unsupported) (`verify.py:206`) |
| Too few deliverables | the request says "three logos" (2 to 20) and fewer files came, unless honest |
| Named file not handed over | the request names a file to make after a making verb ("write leap.py"; `DELIVERABLE`) and no handed-over file has that name, unless honest. A file the harness collected itself is stored as `<persona>_<name>` and counts. After a message the user sent mid-task that names a file, only that message's names are due (`files_from`, [16](16-interjection.md)): a redirect from `fib.py` to `primes.py` no longer asks for `fib.py` |
| Browser claim without browser use | "checked … in the browser" and no `browser.` in the log, unless the log shows `nango.open_page` / `connect_and_wait` / `start` (a page opened in the user's own browser, `OPENED_FOR_USER`) |
| Promise instead of work | "next I'll …" and not a question to the user, nor waiting for the user's word ("Reply go and I'll create it", "once you've saved it, say saved and I'll finish": `USER_FIRST` before the promise in its line) |
| Secret asked for in the chat (nango in use) | the answer asks the user to paste or send a password, key or client secret (`asks_for_secret`) and doesn't say it is asked hidden: `nango.setup()` asks, typed hidden |
| Nango step claimed or promised, not done | `nango` is the topic (in the evidence, request or answer) and no `nango.…(` call is in this reply's log, and the answer says an account "is connected" / "all set up" (`NANGO_CLAIM`) or "I'll hand it to Nango" (`NANGO_PROMISE`) |

The harness also hides the answer to any free-text `ask_user` whose question asks for a password or
key (`asks_for_secret`, used by `Harness._ask_user`), whatever the worker passed: the same check as
the first of the two nango rules, applied to a question instead of an answer.

`work_log` (`learn.py:151`) is the compact log Jev reads: code heads and 350-char outputs, plus
the output lines that mention the answer's numbers and words. `seen_text` (`learn.py:137`) is the
uncut evidence for grounding: everything the worker was shown, but not its own replies.

## File checks (`Verifier._file`, `verify.py:296`)

| Type | Checks |
|---|---|
| any | exists; not empty; its whole content isn't just a file name (`save_artifact("planets.csv", "planets.csv")` once handed over an 11-byte file; `_names_a_file`, `verify.py:122`) |
| `.png .jpg .jpeg .gif .webp` | magic bytes match |
| `.py` | compiles |
| `.json` | parses |
| `.html .htm .svg .md .txt .css .js .csv` | no placeholder text (lorem ipsum, TODO, `[Your Company]`, `{{ x }}`, example.com…); HTML visible text only |
| `.html .htm` | no `src`/`href` to local files that don't exist; renders without script errors or a blank body (`_render`, `verify.py:307`) |

## Risky project changes

```mermaid
flowchart LR
  A["changed files<br/>(project.changed())"] --> B["diff vs .project_backups<br/>change_text()"]
  B --> C{"project tests pass?"}
  C -- no --> P1["problem: tests fail<br/>(last 1.5k of output)"]
  C -- yes --> D["Jev risk per file"]
  D --> E{"any ≥ 0.65?"}
  E -- no --> OK["pass: no risky changes"]
  E -- yes --> F{"severe regex on +/- lines<br/>or Jev severity ≥ 0.20?"}
  F -- yes --> S["strong model reviews"]
  F -- no --> Fa["fast model reviews"]
  S --> R{"NO PROBLEMS?"}
  Fa --> R
  R -- yes --> OK2["pass"]
  R -- no --> P2["problem: review findings<br/>(the worker may dispute them in its answer)"]
```

A diff is reviewed at most once (`Verifier._reviewed`), so a worker that disputes the findings
without changing anything isn't bounced twice for the same diff.
