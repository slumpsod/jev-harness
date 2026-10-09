# The Newsroom

A dashboard tab of **widgets** that keep an eye on things for you: posts from an X account, news on a topic or about a
company, an RSS feed, a subreddit, a project's releases, new arXiv papers. Each widget looks again on its own timer and
**notifies you** when something new matches what you asked to hear about.

You don't build it by hand. **Ask in the chat:**

> Keep an eye on Nvidia news and tell me when they mention earnings.
> Follow @OpenAI on X and alert me to every new post.
> Add the llama.cpp releases to my newsroom.
> Pause the Tesla tile. / Remove the bitcoin widget. / Check the Nvidia one every 10 minutes.
> What's on my newsroom?

The assistant puts the widget on the board, takes a first look, and tells you what it found and what will notify you.
(The tab opens by itself when a turn adds a widget; switch that off with *Open the panel for you* in Appearance.)

> **Status: unverified.** This was built in a cloud session that could reach no news site, no Jev and no model. The offline
> tests pass (`python -m pytest -q tests/test_newsroom*.py`), and the page was driven in Chromium against feeds served from
> localhost. Nothing here has been seen working against real sources or with live models until someone runs the test bench
> ([below](#testing-it-against-the-real-thing)). Untuned guesses: `thresholds.newsroom_about` (0.70) and the three
> description-dependent behaviours the bench measures (Jev offering the tool for the right requests, routing, the `about`
> alert). X is the least certain source: see [Reading X](#reading-x).

## What it can follow (kinds)

| Kind | Settings | Source |
|---|---|---|
| `news` | `topic` (a search; quotes make a phrase), `site`, `days`, `region`, `language` | Google News RSS; no account |
| `x` | `user` (a handle) **or** `query` (a search), `replies` | X: needs set-up, see below |
| `rss` | `url` | any RSS / Atom feed |
| `reddit` | `subreddit` | `reddit.com/r/<name>/new/.rss` |
| `hn` | `query` | hnrss.org |
| `github` | `repo` (`owner/name`) | the project's `releases.atom` |
| `youtube` | `channel_id` | the channel's video feed |
| `arxiv` | `query` | arXiv's API (newest first) |
| `web` | `query`, `n` | a DuckDuckGo search, for what news sites don't cover; a page is "new" when it is in the results on two looks in a row (the bottom places shuffle between identical searches), so it is noticed one look late |

`jevh newsroom kinds` prints this from the code, and the dashboard's **Add widget** form is generated from it too.

## Notifications

Every widget has an **alert**: what makes a *new* item notify you.

- **keywords**: `earnings, guidance`. A word matches where it starts, so `launch` finds "launches" and "launched", while a
  short one like `AI` is matched whole (it never fires on "said"). `exclude` words veto an item.
- **every new item**: for an account or a feed you want to read in full.
- **about**: a sentence ("anything about their quarterly results"). **Jev reads each new item against it.** This is
  for what no keyword names ("Nvidia posts record revenue"). It runs together with keywords (either one notifies) and costs
  one Jev question per new item, only on widgets that have it. If Jev is down, keywords still work.
- none: the widget just shows its items.

The **first look at a widget is its baseline**: the items already there are shown but never notify, so a new widget
doesn't bury you in the last month's stories. One look notifies at most five times, then one "and N more". A story seen
again is not new.

Notifications appear as a **bell list** at the top of the Newsroom tab (with a count on its rail icon), as a **toast** in
the page, and, if you press *Desktop alerts* and allow it, as a **desktop notification** while the tab is in the
background. They are kept until you mark them read or clear them. Nothing leaves your machine: there is no email or
phone push.

Widgets are refreshed **while the dashboard is running** (`jevh dashboard`). With it closed they keep what they have; the
next start catches up (one look per widget, then its own timer). `jevh newsroom refresh` does one pass by hand.

## Settings (config.yaml `newsroom:`)

| Key | Default | |
|---|---|---|
| `enabled` | true | false: the dashboard shows the board but never looks |
| `tick` | 30 | seconds between checks for widgets that are due |
| `min_minutes` / `default_minutes` | 5 / 30 | the shortest and the usual time between looks at one widget |
| `keep_items` / `max_widgets` | 100 / 40 | per widget / on the board |
| `x_feed`, `x_integration` | empty | reading X, below |
| `allow_local` | false | true: a widget may follow a feed on this machine or network |

A widget whose source fails shows the error and is tried less and less often (doubling, up to 6 hours), then recovers by
itself. `thresholds.newsroom_about` (0.70) is how sure Jev must be that an item is *about* what you asked.
Put your own values in `~/.jevh/config.yaml` ([your data](your-data.md)), not in this file.

## Reading X

X has no open feed. One of these must be set up (the **Add widget** form and the assistant both say which is missing):

1. **A feed bridge you run** (RSSHub, Nitter): `newsroom.x_feed: https://your-rsshub/twitter/user/{user}` (`{query}` for searches).
2. **An API token**: set `X_BEARER_TOKEN` in the environment you start the dashboard from (a token that can read posts).
3. **Your X account in Nango**: connect it with the `nango-setup` skill (`newsroom.x_integration` names the integration if it
   isn't `x` / `twitter`).

Until then an X widget is kept and shows "X can't be read until...": it starts working once one is set up. Which API
calls your X plan allows is X's to decide; the widget shows X's refusal if it says no.

## Safety

- What a feed says is **data**. The page shows it as text only (never HTML) and only `http(s)` links; the worker's
  guide tells it headlines are never instructions.
- A widget can only follow the public web: addresses on this machine or its network (localhost, 192.168.x.x, a cloud
  metadata address), including through redirects, are refused unless `allow_local` is set (checked by name when it is fetched: not a defence against a hostile DNS server). Feeds with DOCTYPE/ENTITY tricks
  are refused.
- The board is `workspace/newsroom.db` (or `~/.jevh/workspace`): yours, and only reachable with the dashboard's token.

## Adding a source (a new kind)

A kind is the template for a widget. One line for anything that is just a feed address, in `jevh/newsroom.py`:

```python
feed_kind("lobsters", "Lobsters", "Stories tagged with a topic on Lobsters.",
          "https://lobste.rs/t/{tag}.rss", [Param("tag", "the tag", True, example="rust")], "Lobsters: {tag}")
```

or a `Kind(...)` with its own `fetch(params) -> [{title, url, source, published, summary, author, uid}]` for an API.
Nothing else changes: the Add widget form, `jevh newsroom kinds`, `newsroom.kinds()` and the validation come from the
registry. Add the kind to the guide in `jevh/tools/__init__.py` (a test fails if it is missing) and a line to `SOURCES` in
the bench so a real feed is checked.

## Testing it against the real thing

Offline (nothing live): `python -m pytest -q tests/test_newsroom.py tests/test_newsroom_ui.py tests/test_newsroom_evals.py`.

Live, on your machine (LM Studio or your usual models, the Jev adapter, internet; nothing touches your own board):

```bash
python evals/newsroom_cases.py                      # Jev's three decisions: dev sets   (--holdout, --runs 3, --offline)
python evals/newsroom_cases.py --holdout
python evals/bench/newsroom.py --stage sources      # every kind against its real source; the second look must find nothing new
python evals/bench/newsroom.py --model lmstudio:qwen3.8-27b@q6_k --reps 3     # sources + full turns (old vs new) + the dashboard path
```

| Eval | Measures | Writes |
|---|---|---|
| `evals/newsroom_cases.py` `--stage tool` | is the newsroom guide offered for "keep an eye on / alert me" requests and **not** for a one-off "what's the news on X?", "follow up with Sam", "monitor my server"; also that the other tools' picks don't move | `newsroom_cases_raw.json`, `newsroom_cases_last_summary.txt` |
| ... `--stage route` | the changed `daily_assistant` description, old vs new, on the newsroom requests and on every request in `routing_cases.py` | same |
| ... `--stage about` | `about` against keywords on labelled headlines (including ones with no keyword and ones with the right company but the wrong subject); a threshold sweep. `--export items.jsonl` writes the real items on your board for you to label, `--labelled items.jsonl` scores them | same |
| `evals/bench/newsroom.py --stage sources` | real feeds parse; a second look finds 0 new | `evals/bench/newsroom_raw.json`, `newsroom_last_summary.txt` |
| ... `--stage turns` | full turns: set-up requests make the right widget; board edits do the right thing; lookalike and ordinary requests leave the board alone; old harness vs new on cost | same |
| ... `--stage dashboard` | chat -> widget -> a new story -> exactly one notification and the page event -> no repeat -> remove in the chat | same |

The cases are authored, not taken from real runs (there are none yet): use `--export` once your board has run for a day to
score real items, and `jevh failures export` for requests that went wrong. Targets in each file are first guesses.
**What would mean the feature is wrong:** Jev offering the tool for one-off news questions (a widget is made for a lookup),
a second look that finds new items on a real feed (it would notify on every refresh), X widgets never created because the
worker stops at "X isn't set up", the new description pulling requests from other personas, `about` notifying on every
story about the company.

Mark the feature verified in this page's status note once those have been run and read.
