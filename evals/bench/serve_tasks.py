"""Coding tasks for evals/bench/serve_compare.py: a request, optional starter files, and a HIDDEN pytest file.

Each task is run through the harness (high_end_dev) in a fresh folder; afterwards the hidden tests run against what
the worker wrote. Score = share of hidden tests passed, "pass" = all of them. `{dir}` is the task's folder.
The tests are checked against reference solutions (evals/bench/serve_tasks_ref/) by tests/test_serve_compare.py.
"""
from __future__ import annotations

TASKS: dict[str, dict] = {}


def task(tid: str, prompt: str, tests: str, files: dict[str, str] | None = None) -> None:
    TASKS[tid] = {"id": tid, "prompt": prompt, "tests": tests, "files": files or {}}


# --------------------------------------------------------------------------------------------------------------
task("ledger", """Write {dir}/ledger.py with a function summarize(path) that reads a bank-statement CSV and returns monthly totals.

The CSV has the header `date,description,amount,currency`. Rules:
- date is either YYYY-MM-DD or DD/MM/YYYY (day first). Group by month as "YYYY-MM".
- amount may look like `1,234.50`, `$12.00`, `-12.5`, or `(45.00)` (parentheses mean negative). A blank amount row is skipped.
- Positive amounts are "in", negative ones are "out" (stored as a positive number). net = in - out.
- A row that is an exact duplicate of an earlier row (all four fields identical) is counted once.
- Return a dict {"YYYY-MM": {"in": float, "out": float, "net": float}} with the months in ascending order, all numbers rounded to 2 decimals.
Check it against a CSV you make yourself, then call final().""",
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ledger import summarize

def w(tmp_path, rows):
    p = tmp_path / "s.csv"
    p.write_text("date,description,amount,currency\\n" + "\\n".join(rows) + "\\n", encoding="utf-8")
    return str(p)

def test_basic(tmp_path):
    r = summarize(w(tmp_path, ["2024-01-05,pay,2000.00,USD", "2024-01-09,rent,-800.00,USD"]))
    assert r == {"2024-01": {"in": 2000.0, "out": 800.0, "net": 1200.0}}

def test_formats(tmp_path):
    r = summarize(w(tmp_path, ['15/02/2024,"big, pay","1,234.50",USD', "2024-02-03,x,$12.00,USD", "2024-02-04,y,(45.00),USD", "2024-02-05,z,-12.5,USD"]))
    assert r["2024-02"] == {"in": 1246.5, "out": 57.5, "net": 1189.0}

def test_day_first(tmp_path):
    r = summarize(w(tmp_path, ["03/04/2024,a,10,USD"]))
    assert list(r) == ["2024-04"]

def test_blank_and_duplicates(tmp_path):
    r = summarize(w(tmp_path, ["2024-03-01,a,5,USD", "2024-03-01,a,5,USD", "2024-03-01,a,5.00,USD", "2024-03-02,b,,USD"]))
    assert r["2024-03"]["in"] == 10.0

def test_sorted_months(tmp_path):
    r = summarize(w(tmp_path, ["2024-12-01,a,1,USD", "2024-02-01,a,1,USD", "01/11/2023,a,1,USD"]))
    assert list(r) == ["2023-11", "2024-02", "2024-12"]

def test_rounding(tmp_path):
    r = summarize(w(tmp_path, ["2024-05-01,a,0.1,USD", "2024-05-02,b,0.2,USD", "2024-05-03,c,-0.05,USD"]))
    assert r["2024-05"] == {"in": 0.3, "out": 0.05, "net": 0.25}
""")

# --------------------------------------------------------------------------------------------------------------
task("ttlcache", """Write {dir}/ttlcache.py with a class TTLCache(capacity, ttl, clock=time.monotonic).

- put(key, value, ttl=None): store the value; ttl overrides the default for this entry. Putting an existing key replaces its value and TTL and makes it the most recently used.
- get(key, default=None): the value, or default if missing or expired. A successful get makes the key the most recently used but does NOT extend its TTL.
- An entry is expired when clock() - stored_at >= its ttl (so it expires exactly at the TTL). Expired entries are removed when seen.
- When a put would exceed capacity, first remove all expired entries; if it is still over capacity, evict the least recently used entry.
- delete(key) -> True if it was there and live, else False.
- purge() -> number of expired entries removed.
- keys() -> list of live keys from least to most recently used.
- len(cache) counts live entries only.
`clock` is injected so tests can fake time. Check it with a fake clock, then call final().""",
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ttlcache import TTLCache

class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t

def test_basic():
    c = TTLCache(2, 10, Clock())
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1 and c.get("zz", 5) == 5 and len(c) == 2

def test_lru_eviction():
    c = TTLCache(2, 10, Clock())
    c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
    assert c.keys() == ["a", "c"] and c.get("b") is None

def test_expires_exactly_at_ttl():
    k = Clock(); c = TTLCache(3, 5, k)
    c.put("a", 1); k.t = 4.99
    assert c.get("a") == 1
    k.t = 5.0
    assert c.get("a") is None and len(c) == 0

def test_get_does_not_extend_ttl():
    k = Clock(); c = TTLCache(3, 5, k)
    c.put("a", 1); k.t = 3; c.get("a"); k.t = 5
    assert c.get("a") is None

def test_put_resets_ttl_and_recency():
    k = Clock(); c = TTLCache(3, 5, k)
    c.put("a", 1); c.put("b", 2); k.t = 4; c.put("a", 9)
    assert c.keys() == ["b", "a"]
    k.t = 6
    assert c.get("b") is None and c.get("a") == 9

def test_per_entry_ttl():
    k = Clock(); c = TTLCache(3, 100, k)
    c.put("a", 1, ttl=2); c.put("b", 2); k.t = 2
    assert c.get("a") is None and c.get("b") == 2

def test_expired_removed_before_lru_eviction():
    k = Clock(); c = TTLCache(2, 100, k)
    c.put("old", 1, ttl=1); c.put("keep", 2); c.get("old"); k.t = 1
    c.put("new", 3)
    assert c.keys() == ["keep", "new"]

def test_delete_and_purge():
    k = Clock(); c = TTLCache(5, 3, k)
    c.put("a", 1); c.put("b", 2, ttl=10); c.put("c", 3)
    assert c.delete("a") is True and c.delete("a") is False
    k.t = 3
    assert c.delete("c") is False
    c.put("d", 4)
    assert c.purge() == 0
    c.put("e", 5, ttl=1); k.t = 4
    assert c.purge() == 1 and c.keys() == ["b", "d"]
""")

# --------------------------------------------------------------------------------------------------------------
task("mdtable", """Write {dir}/mdtable.py with a function parse_table(text) that returns the FIRST GitHub-style markdown table in text as a list of dicts.

- The header row gives the keys (cells stripped of surrounding spaces). The next row must be the separator (cells made of dashes with optional colons, e.g. `---`, `:--`, `--:`, `:-:`); a pipe line that is not followed by a separator row is not a table.
- Leading and trailing pipes are optional. Cells are split on `|` but `\\|` is an escaped pipe that stays inside the cell (as a plain `|`).
- Cell values are stripped. A value that is an integer (like `42` or `-7`) becomes an int, a decimal (like `3.5`) a float, an empty cell None, anything else a str.
- A row with fewer cells than the header is padded with None; extra cells are dropped.
- The table ends at the first blank line or the first line without a pipe. Return [] if there is no table.
Check it on a few examples, then call final().""",
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mdtable import parse_table

def test_basic():
    t = "| name | qty |\\n|---|---|\\n| apple | 3 |\\n| pear | 4 |\\n"
    assert parse_table(t) == [{"name": "apple", "qty": 3}, {"name": "pear", "qty": 4}]

def test_no_outer_pipes_and_alignment():
    t = "a | b | c\\n:-- | :-: | --:\\n1 | 2.5 | x"
    assert parse_table(t) == [{"a": 1, "b": 2.5, "c": "x"}]

def test_escaped_pipe():
    t = "| k | v |\\n|---|---|\\n| a\\\\|b | c |"
    assert parse_table(t) == [{"k": "a|b", "v": "c"}]

def test_empty_and_short_and_long_rows():
    t = "| a | b | c |\\n|---|---|---|\\n| 1 | | 3 |\\n| 4 |\\n| 5 | 6 | 7 | 8 |"
    assert parse_table(t) == [{"a": 1, "b": None, "c": 3}, {"a": 4, "b": None, "c": None}, {"a": 5, "b": 6, "c": 7}]

def test_first_table_only_and_text_around():
    t = "Intro | not a table\\n\\n| x |\\n|---|\\n| 1 |\\n\\ntext\\n\\n| y |\\n|---|\\n| 2 |\\n"
    assert parse_table(t) == [{"x": 1}]

def test_ends_at_line_without_pipe():
    t = "| x |\\n|---|\\n| 1 |\\nplain line\\n| 2 |"
    assert parse_table(t) == [{"x": 1}]

def test_pipe_line_without_separator_is_not_a_table():
    assert parse_table("| a | b |\\n| 1 | 2 |\\n") == []
    assert parse_table("just text") == []

def test_negative_numbers_and_text():
    t = "| v |\\n|---|\\n| -7 |\\n| -2.25 |\\n| abc |"
    got = [r["v"] for r in parse_table(t)]
    assert got == [-7, -2.25, "abc"] and isinstance(got[0], int) and isinstance(got[1], float)
""")

# --------------------------------------------------------------------------------------------------------------
task("freeslot", """Write {dir}/freeslot.py with a function first_free(busy, duration, day_start, day_end).

- busy: a list with one entry per person; each entry is a list of (start, end) busy intervals as "HH:MM" strings (24-hour). Intervals may be unsorted and may overlap or touch.
- duration: meeting length in minutes (a positive int). day_start, day_end: "HH:MM" bounds; the meeting must fit completely inside them.
- Return the earliest start time "HH:MM" such that the meeting [start, start+duration) overlaps nobody's busy time, or None if there is no such time. Two intervals that only touch (one ends exactly when the other starts) do not overlap. Start times can be any minute, not only multiples of 5.
Check it with a few cases (including touching intervals and no-fit), then call final().""",
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from freeslot import first_free

def test_empty_calendar():
    assert first_free([], 30, "09:00", "17:00") == "09:00"
    assert first_free([[]], 30, "09:00", "17:00") == "09:00"

def test_simple_gap():
    assert first_free([[("09:00", "10:00")], [("10:30", "11:00")]], 30, "09:00", "17:00") == "10:00"

def test_touching_is_free():
    assert first_free([[("09:00", "10:00")], [("10:30", "12:00")]], 30, "09:00", "17:00") == "10:00"
    assert first_free([[("09:00", "10:00")], [("10:15", "12:00")]], 30, "09:00", "17:00") == "12:00"

def test_unsorted_overlapping():
    busy = [[("13:00", "14:00"), ("09:00", "09:45"), ("09:30", "11:00")], [("11:00", "12:30")]]
    assert first_free(busy, 60, "09:00", "17:00") == "14:00"

def test_exact_fit_at_end_and_overflow():
    assert first_free([[("09:00", "16:00")]], 60, "09:00", "17:00") == "16:00"
    assert first_free([[("09:00", "16:01")]], 60, "09:00", "17:00") is None

def test_busy_outside_window_and_start_inside_busy():
    assert first_free([[("07:00", "09:30")]], 15, "09:00", "10:00") == "09:30"
    assert first_free([[("08:00", "18:00")]], 15, "09:00", "17:00") is None

def test_odd_minutes():
    assert first_free([[("09:00", "09:07")], [("09:07", "09:52")]], 10, "09:00", "11:00") == "09:52"

def test_duration_longer_than_window():
    assert first_free([], 600, "09:00", "17:00") is None
""")

# --------------------------------------------------------------------------------------------------------------
BUGGY = '''"""Small text statistics helpers."""
import re


def tokenize(text):
    """Lowercase words. Letters and digits make words; an apostrophe inside a word stays (don't -> don't);
    everything else separates words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def top_n(counts, n):
    """The n most common (word, count) pairs: highest count first, ties broken alphabetically (a before b)."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]), reverse=True)[:n]


def sentences(text):
    """Split into sentences at . ! or ? followed by whitespace and a capital letter or digit.
    Abbreviations like Dr., Mr., e.g. and i.e. do not end a sentence."""
    parts = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9])", text.strip())
    return [p for p in parts if p]


def wrap(text, width):
    """Greedy word wrap: lines of at most `width` characters, words kept whole (a word longer than width gets
    its own line). Returns a list of lines."""
    lines, cur = [], ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) < width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
'''

VISIBLE = '''import textstats


def test_tokenize_basic():
    assert textstats.tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_tokenize_apostrophe():
    assert textstats.tokenize("Don't stop") == ["don't", "stop"]


def test_top_n_ties():
    assert textstats.top_n({"b": 2, "a": 2, "c": 5}, 2) == [("c", 5), ("a", 2)]
'''

task("bugfix", """The folder {dir} has textstats.py (a few text helpers) and test_visible.py. Some of the tests fail and the module has other \
bugs that the visible tests do not catch. Read the docstrings (they are the spec), fix textstats.py so every function does what its \
docstring says, run the tests, and call final(). Do not change the function names or signatures.""",
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import textstats as T

def test_tokenize():
    assert T.tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert T.tokenize("Don't stop; it's 5 o'clock.") == ["don't", "stop", "it's", "5", "o'clock"]
    assert T.tokenize("'quoted' words") == ["quoted", "words"]
    assert T.tokenize("") == []

def test_top_n():
    assert T.top_n({"b": 2, "a": 2, "c": 5}, 2) == [("c", 5), ("a", 2)]
    assert T.top_n({"x": 1, "y": 1, "z": 1}, 5) == [("x", 1), ("y", 1), ("z", 1)]
    assert T.top_n({}, 3) == []

def test_sentences_basic():
    assert T.sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]
    assert T.sentences("It was 5. 6 came next.") == ["It was 5.", "6 came next."]

def test_sentences_abbreviations():
    assert T.sentences("Dr. Smith met Mr. Jones. They talked.") == ["Dr. Smith met Mr. Jones.", "They talked."]
    assert T.sentences("Use fruit, e.g. Apples. Done.") == ["Use fruit, e.g. Apples.", "Done."]
    assert T.sentences("That is, i.e. Nothing. Fine.") == ["That is, i.e. Nothing.", "Fine."]

def test_sentences_no_split_lowercase():
    assert T.sentences("pi is 3.14 and so on. then more") == ["pi is 3.14 and so on. then more"]

def test_wrap_exact_width():
    assert T.wrap("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]
    assert T.wrap("aaa bbb ccc", 11) == ["aaa bbb ccc"]
    assert T.wrap("aaa bbb ccc", 10) == ["aaa bbb", "ccc"]

def test_wrap_long_word_and_empty():
    assert T.wrap("a verylongword b", 4) == ["a", "verylongword", "b"]
    assert T.wrap("", 10) == []
    assert T.wrap("  spaced   out  ", 20) == ["spaced out"]
""", files={"textstats.py": BUGGY, "test_visible.py": VISIBLE})

# --------------------------------------------------------------------------------------------------------------
task("todocli", """Write a command-line to-do tool at {dir}/todo.py (standard library only, argparse).

Usage: `python todo.py [--file PATH] COMMAND ...` where PATH (default `todo.json` in the current directory) is a JSON file the tool creates when needed.
Commands:
- `add TEXT...` (the words are joined with spaces): adds an item with the next id (ids start at 1 and are never reused, even after rm) and prints `added N`.
- `list [--all]`: prints one line per item `N [ ] text` (open) or `N [x] text` (done), ordered by id; without --all only open items are shown. Prints nothing if there are none.
- `done N`: marks item N done and prints `done N`.
- `rm N`: removes item N and prints `removed N`.
For `done` or `rm` with an id that does not exist, print `no such item: N` to stderr and exit with status 1. Everything else exits 0.
Try it from the shell, then call final().""",
"""
import subprocess, sys, os, json
HERE = os.path.dirname(__file__)

def run(tmp_path, *args, check=None):
    p = subprocess.run([sys.executable, os.path.join(HERE, "todo.py"), "--file", str(tmp_path / "t.json"), *args],
                       capture_output=True, text=True, cwd=str(tmp_path))
    return p

def test_add_and_list(tmp_path):
    assert run(tmp_path, "add", "buy", "milk").stdout.strip() == "added 1"
    assert run(tmp_path, "add", "call mom").stdout.strip() == "added 2"
    assert run(tmp_path, "list").stdout.splitlines() == ["1 [ ] buy milk", "2 [ ] call mom"]

def test_done_hides_unless_all(tmp_path):
    run(tmp_path, "add", "a"); run(tmp_path, "add", "b")
    assert run(tmp_path, "done", "1").stdout.strip() == "done 1"
    assert run(tmp_path, "list").stdout.splitlines() == ["2 [ ] b"]
    assert run(tmp_path, "list", "--all").stdout.splitlines() == ["1 [x] a", "2 [ ] b"]

def test_rm_and_ids_not_reused(tmp_path):
    run(tmp_path, "add", "a"); run(tmp_path, "add", "b")
    assert run(tmp_path, "rm", "2").stdout.strip() == "removed 2"
    assert run(tmp_path, "add", "c").stdout.strip() == "added 3"
    assert run(tmp_path, "list", "--all").stdout.splitlines() == ["1 [ ] a", "3 [ ] c"]

def test_missing_id(tmp_path):
    run(tmp_path, "add", "a")
    for cmd in ("done", "rm"):
        p = run(tmp_path, cmd, "9")
        assert p.returncode == 1 and "no such item: 9" in p.stderr and p.stdout.strip() == ""

def test_empty_list_and_exit_codes(tmp_path):
    p = run(tmp_path, "list")
    assert p.returncode == 0 and p.stdout == ""
    assert run(tmp_path, "add", "x").returncode == 0

def test_persists_as_json_file(tmp_path):
    run(tmp_path, "add", "persist me")
    data = json.loads((tmp_path / "t.json").read_text())
    assert "persist me" in json.dumps(data)

def test_default_file_in_cwd(tmp_path):
    p = subprocess.run([sys.executable, os.path.join(HERE, "todo.py"), "add", "x"], capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 0 and (tmp_path / "todo.json").exists()
""")
