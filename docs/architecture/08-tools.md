# 08 · Tools and their own Jev loops

Tools are Python modules imported into a worker's REPL under their name (`init` op,
`repl/server.py:364`). Jev picks per task which tools' **guides** go into the prompt (step
`tools`). All of the persona's tools are importable either way. The tools run inside the REPL
process. Several of them run their own Jev loops, talking straight to the adapter
([02](02-jev-decisions.md#tool-side-decisions-inside-the-repl-recorded-as-kind-jev_tools)).

| Tool | Module | Talks to | Jev loop | Harness callbacks |
|---|---|---|---|---|
| `browser` | `tools/browser.py` | Chromium via Playwright (own thread) | `do()`: pick element / value / option, button checks, done-checks | `look` → the worker's model sees a screenshot |
| `web` | `tools/web.py` | DuckDuckGo HTML, any page | result ranking, passage picking | — |
| `project` | `tools/project.py` + `tools/rules.py` | the user's project folder, shell | `find` / `find_files` ranking; rule enforcer on every edit | — |
| `nango` | `tools/nango.py` | Nango API proxy → the user's apps | — | `ask_user` for every change |
| `comfy` | `tools/comfy.py` | ComfyUI `:8188` | — | — |
| MCP | `mcp.py` (in the harness) | stdio JSON-RPC servers | `mcp_rank` / `mcp_fits` at task setup | `mcp(key, **args)` callback |

## browser.do(goal): the Jev-driven browser loop

The worker states **one outcome**, and Jev picks every click. The worker's model never reads a
page snapshot. Each call runs on the browser's own thread (`_run`, `tools/browser.py:1283`).

```mermaid
flowchart TD
  S["do(goal, values, max_actions=8)<br/>tools/browser.py:837"] --> O["observe(): settle, list elements,<br/>wait while a new page grows"]
  O --> BW{"bot wall text?<br/>(regex)"}
  BW -- yes --> R1["blocked"]
  BW -- no --> AF{"a given value obviously<br/>matches an empty field?"}
  AF -- yes --> FILL["type it (no Jev)"] --> O
  AF -- no --> Q["Jev choice Q_TARGET over elements<br/>+ scroll / back / none<br/>extras: done, error, blocked, login<br/>(> 16 options: Q_GROUP first)"]
  Q --> D{"done ≥ 0.9?<br/>(capped at 0.29 if the goal says<br/>save/submit and nothing was sent)"}
  D -- yes --> R2["done"]
  D -- no --> E{"error ≥ .7 / blocked ≥ .85 /<br/>login ≥ .7?"}
  E -- yes --> R3["error / blocked / needs_login"]
  E -- no --> N{"best = none?"}
  N -- yes --> R4["done / likely_done / stuck"]
  N -- no --> AMB{"best < 0.2?"}
  AMB -- yes --> R5["ambiguous + candidates"]
  AMB -- no --> K{"element kind"}
  K -- "field" --> V["Jev Q_VALUE: which value<br/>(or quoted text in the goal)"]
  V -- missing --> R6["needs_value"]
  K -- "dropdown" --> OP["Jev: which option"]
  K -- "button" --> BT["Jev wanted / irreversible / pending"]
  BT -- "not wanted or form not ready" --> DEF["defer this button"] --> O
  BT -- irreversible --> R7["needs_confirmation<br/>(retry with allow_irreversible=True<br/>after ask_user)"]
  K -- "link / other" --> ACT
  V --> ACT
  OP --> ACT
  BT -- ok --> ACT
  ACT["act via Playwright<br/>(outlined and paused if watched)"] --> CH{"page changed?"}
  CH -- "no (3×)" --> R8["stuck"]
  CH -- "search thrown away" --> R1
  CH -- yes --> O
```

Every result is `{status, goal, url, title, actions, note, page (1.2k), p, jev_calls,
jev_seconds, seconds}`. When Jev couldn't finish, it adds `candidates` (numbered elements), so the
worker can act itself with `browser.click(i)` / `fill(i, text)` / `select(i, option)`.

`browser.test(url, steps, expect)` runs `do()` per step with `act_first`, then checks `expect`
with two nouls (`shows ≥ .55` and `holds ≥ .6`) and reports JS and console errors. With the
dashboard's "Watch browser" on, the browser runs with a CDP port (`JEVH_BROWSER_DOCK`), and the
dashboard mirrors it and passes the user's clicks back ([10](10-observability.md)).

## web.ask(question)

```mermaid
sequenceDiagram
  participant W as worker code
  participant WEB as web.ask (REPL)
  participant DDG as DuckDuckGo
  participant J as Jev (direct)
  participant P as pages (parallel)
  W->>WEB: ask(question)
  WEB->>DDG: search(question, 8)
  DDG-->>WEB: title, url, snippet
  WEB->>J: choice "which result answers question" (choose)
  Note over WEB: best first, one page per site, pages + 2 spares
  par up to 4 pages
    WEB->>P: fetch (10s timeout, HTML→text, PDF via pypdf)
    WEB->>J: pick_passages: keyword prefilter to 48, choice, noul on top 3 ≥ .3
  end
  WEB-->>W: {passages: [{title, url, text ≤ 2.5k}], results, unreadable?}
```

## project: edits go through the rule enforcer

```mermaid
sequenceDiagram
  participant W as worker code
  participant PR as project.edit / write
  participant RU as rules.check
  participant J as Jev (direct)
  participant FS as project folder
  W->>PR: edit(path, old, new, rule_ok=None)
  PR->>PR: _path(): must stay inside the project
  PR->>FS: _backup(): first touch copies the original to .project_backups/
  PR->>RU: check(root, rel, old, new, existed)
  Note over RU: rules = bullets of RULES.md / AGENTS.md / CLAUDE.md / .jevh/rules.md<br/>scoped by backticked globs (plain code)
  alt a read-only scoped rule applies
    RU-->>PR: Violation (no Jev call)
  else up to 12 rules in scope
    RU->>J: noul "does change to file break rule?" per rule
    J-->>RU: P(breaks)
  end
  alt P ≥ rule_block 0.8 and no rule_ok
    PR-->>W: raise RuleViolation (rule quoted)
  else rule_ok given
    PR->>PR: record override (reported in final answer)
    PR->>FS: write
  else
    PR->>FS: write
  end
```

`project.changed()` lists the touched files. `final()` sends that list to the gate, which runs
the project's tests and reviews risky diffs against the backups ([07](07-final-gate.md)).

## nango: changes are approved by the user, from inside the tool

`nango.request` (`tools/nango.py:252`) sends every call through Nango's `/proxy` with the
connection's key, so no token reaches a prompt. A method that isn't a read (`is_read`), or any
call with `base_url=`, goes through `_approve` (`tools/nango.py:225`). `_approve` calls the REPL's
own `_call("ask_user", …)` directly, so the worker's code can't answer for the user. The options
are `Allow`, `Allow all <app> changes this task`, and `Don't allow`. `tools.nango_writes: deny`
refuses all changes, and `allow` skips the question.

## MCP (`mcp.py`)

`McpHub` starts each configured server on first use (`initialize`, `tools/list`), and the servers'
tools form one catalog keyed `server.tool`. At task setup Jev picks at most 3 of them
(`mcp_rank`, `mcp_fits`), and their signatures go in the prompt. The worker calls
`mcp("server.tool", arg=…)` → callback → `_mcp_call` checks the persona's `mcp` globs →
`McpServer.call` → `tools/call` JSON-RPC → text content.
