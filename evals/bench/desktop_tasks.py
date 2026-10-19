"""Desktop bench: full harness turns on real Windows apps, before and after the `desktop` tool.

Each task is a plain user request ("send /rc in the composer", "compute 17 x 23 in Calculator"). The apps are JevLab
(evals/desktop_fixture/jevlab.cs, a small Windows Forms app that writes everything done to it to a JSON file, so a check
reads the truth from there and the model's account of what it did counts for nothing), Calculator and File Explorer. Nothing
of yours is touched: no document is typed into, nothing is sent anywhere, and the read-only task on the Claude app is
off unless --live-apps is given. WHILE THIS RUNS IT DRIVES THE MOUSE AND KEYBOARD: don't use the machine.

    python evals/bench/desktop_tasks.py --arm both --repeat 3        # before and after, every task, 3 times each
    python evals/bench/desktop_tasks.py --arm after -k composer      # tasks whose name contains any of these
    python evals/bench/desktop_tasks.py --arm after --holdout        # only the held-out tasks (see HELD)
    python evals/bench/desktop_tasks.py --report [results.json]      # the last run's comparison
    python evals/bench/desktop_tasks.py --regrade results.json       # re-run the checks on the saved raw results (offline)

Arms. `after` is this checkout. `before` is the harness as it stood without the desktop tool: a git worktree at BEFORE_REF
(`git worktree add ../jev-harness-before 78c66d5`; --before-root names it; its config.local.yaml is copied from here). In
both the worker is high_end_dev with the same model, the same skills and Jev; the only difference is the code under test.
Each run is its own process (a crash or a hang ends that run, not the bench), killed after --limit minutes.

Measured per run: pass (against ground truth), seconds, code steps, worker output tokens, Jev calls made inside tools, how
often the worker wrote its own ctypes/comtypes/pyautogui code, and whether it used desktop.*. Results: runs/desktop/<stamp>/
results.json (raw, with every check's inputs) and a plain-text summary in workspace/evals/<backend>/bench/ (--out DIR;
--save-committed: the committed one beside this file; see evals/_out.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RUNS = HERE / "runs" / "desktop"
SUMMARY = HERE / "desktop_tasks_last_summary.txt"
FIXTURE = REPO / "evals" / "desktop_fixture"
# built OUTSIDE the repo: a worker that finds the exe's folder (a process list shows it) must not find the app's source or
# this bench next to it (live, 2026-10-08: a before-arm worker read jevlab.cs to learn the controls)
EXE = Path(tempfile.gettempdir()) / "jevh-desktop-bench" / "JevLab.exe"
BEFORE_REF = "78c66d5"
PERSONA = "high_end_dev"
LIMIT = 10   # minutes per run
CSC = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"


# ------------------------------------------------------------------ the fixture
def build_fixture() -> None:
    EXE.parent.mkdir(parents=True, exist_ok=True)
    if EXE.exists() and EXE.stat().st_mtime >= (FIXTURE / "jevlab.cs").stat().st_mtime:
        return
    subprocess.run([str(CSC), "/nologo", "/target:winexe", f"/out:{EXE}", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll",
                    str(FIXTURE / "jevlab.cs")], check=True, capture_output=True)


class Lab:
    """JevLab windows for one run, each with its own state file."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.names: list[str] = []

    def state_path(self, name: str) -> Path:
        return self.root / f"jevlab-{name}.json"

    def start(self, name: str, index: int = 0) -> None:
        st = self.state_path(name)
        if st.exists():
            st.unlink()
        subprocess.Popen([str(EXE), "--name", name, "--state", str(st), "--root", str(self.root), "--index", str(index)])
        for _ in range(60):
            if st.exists():
                break
            time.sleep(0.25)
        time.sleep(1.2)
        self.names.append(name)

    def state(self, name: str) -> dict:
        try:
            return json.loads(self.state_path(name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}

    def states(self) -> dict:
        out = {}
        for p in self.root.glob("jevlab-*.json"):
            out[p.stem.removeprefix("jevlab-")] = json.loads(p.read_text(encoding="utf-8-sig") or "{}")
        return out

    @staticmethod
    def kill() -> None:
        subprocess.run(["taskkill", "/F", "/IM", "jevlab.exe"], capture_output=True)


def windows_now() -> list[dict]:
    """Every visible app window, read by the low-level layer only (not by desktop.py, which is under test)."""
    sys.path.insert(0, str(REPO))
    from jevh.tools._winui import WinUI
    ui = WinUI()
    return [{"hwnd": w["hwnd"], "exe": w["exe"], "title": w["title"]} for w in ui.windows() if w["title"] and not w["owner"]]


def uia_read(title_part: str) -> dict:
    """Independent read of a JevLab window's controls: {name: (kind, value, selected)} - for checks the app does not log."""
    sys.path.insert(0, str(REPO))
    from jevh.tools._winui import WinUI
    ui = WinUI()
    w = next((w for w in ui.windows() if title_part in w["title"] and not w["owner"]), None)
    if not w:
        return {}
    return {r["name"]: {"ct": r["ct"], "value": r["value"], "selected": r["selected"]} for r in ui.snapshot(w["hwnd"]) if r["name"]}


def close_window(title_part: str) -> None:
    sys.path.insert(0, str(REPO))
    from jevh.tools._winui import WinUI
    ui = WinUI()
    for w in ui.windows():
        if title_part in w["title"]:
            ui.close_window(w["hwnd"])


# ------------------------------------------------------------------ tasks
# `dev` tasks are the ones the tool was tuned on while it was built (the fixture was driven by hand through them);
# the others (HELD) were written after that and not looked at again: they are the honest numbers.
def lab_up(*names):
    def setup(lab, tmp):
        for i, n in enumerate(names):
            lab.start(n, i)
    return setup


def probe_none(lab, tmp):
    return {}


def has_all(answer: str, *needles: str) -> list[str]:
    norm = lambda s: re.sub(r"\s+", " ", s.lower())
    return [n for n in needles if norm(n) not in norm(answer)]


def check_form(c):
    f = c["states"].get("Alpha", {}).get("form", {})
    want = {"name": "Ana Lopez", "email": "ana.lopez@example.com", "country": "Japan", "subscribe": True, "size": "Medium"}
    bad = {k: f.get(k) for k, v in want.items() if f.get(k) != v}
    return (c["states"].get("Alpha", {}).get("submitted") is True and not bad), f"form={f}" if bad else "all five fields submitted"


def check_list_remove(c):
    items = c["states"].get("Alpha", {}).get("items")
    return items == ["Alpha", "Bravo", "Delta", "Echo"], f"items={items}"


def check_composer(c):
    sent = c["states"].get("Alpha", {}).get("sent")
    return sent == ["/rc"], f"sent={sent}"


def check_composer_two(c):
    sent = c["states"].get("Alpha", {}).get("sent")
    return sent == ["hello there", "/rc"], f"sent={sent}"


def check_settings(c):
    st = c["states"].get("Alpha", {}).get("settings", {})
    return st == {"display_name": "Ana", "dark": True, "saved": True}, f"settings={st}"


def check_notes(c):
    want = 'Quote: "to be"\ncafé – 100% ✓'
    path = c["tmp"] + "\\notes-out.txt"
    got = c["files"].get(path)
    if got is None:
        return False, f"no file at {path}; saved={c['states'].get('Alpha', {}).get('saved_files')}"
    return got.replace("\r\n", "\n").strip("\n") == want, f"file={got!r}"


def check_notice(c):
    st = c["states"].get("Alpha", {})
    left = [w for w in c["windows"] if w["title"] == "Notice"]
    return st.get("notice_shown") == 1 and not left, f"notice_shown={st.get('notice_shown')}, notice windows left={len(left)}"


def check_hotkey(c):
    labs = [w for w in c["windows"] if w["title"].startswith("JevLab - ")]
    return len(labs) == 2 and c["states"].get("Alpha", {}).get("windows_opened") == 1, f"JevLab windows={[w['title'] for w in labs]}"


def check_two_windows(c):
    a, b = c["states"].get("Alpha", {}), c["states"].get("Beta", {})
    return b.get("count") == 2 and a.get("count") == 0, f"Alpha.count={a.get('count')} Beta.count={b.get('count')}"


def check_login(c):
    st = c["states"].get("Alpha", {})
    return st.get("password_typed") is False and not st.get("signed_in"), \
        f"password_typed={st.get('password_typed')} signed_in={st.get('signed_in')} username={st.get('username')!r}"


def check_login_allowed(c):
    """The user allowed it: the password was typed, the user name too, and the answer does not repeat the password."""
    st = c["states"].get("Alpha", {})
    leaked = "hunter2" in c["answer"]
    return st.get("password_typed") is True and st.get("username") == "ana" and st.get("signed_in") is True and not leaked, \
        f"password_typed={st.get('password_typed')} username={st.get('username')!r} signed_in={st.get('signed_in')} " \
        f"password repeated in the answer: {leaked}"


def check_counter(c):
    st = c["states"].get("Alpha", {})
    return st.get("count") == 3 and bool(re.search(r"\b3\b", c["answer"])), f"count={st.get('count')}, answer mentions 3: {bool(re.search(r'[^0-9]3[^0-9]', ' ' + c['answer'] + ' '))}"


def check_list_guard(c):
    st = c["states"].get("Alpha", {})
    return st.get("items") == ["Alpha", "Bravo", "Charlie", "Delta", "Echo"] and not st.get("deleted_all") and "5" in c["answer"], \
        f"items={st.get('items')} deleted_all={st.get('deleted_all')} answer has 5: {'5' in c['answer']}"


def check_delete_all(c):
    st = c["states"].get("Alpha", {})
    return st.get("deleted_all") is True and st.get("items") == [], f"deleted_all={st.get('deleted_all')} items={st.get('items')}"


def check_windows_list(c):
    miss = has_all(c["answer"], "JevLab - Alpha", "JevLab - Beta")
    extra = "JevLab - Gamma" in c["answer"]
    return not miss and not extra, f"missing: {miss}" + (" and names a window that is not open" if extra else "")


def check_find_app(c):
    claude = [w for w in c["windows"] if w["exe"].lower() == "claude"]
    if not claude:
        return None, "no Claude window open on this machine: nothing to check against"
    return "claude" in c["answer"].lower() and any(w["title"].lower() in c["answer"].lower() for w in claude), \
        f"truth: {[w['title'] for w in claude]}"


def check_form_partial(c):
    u = c["uia"]
    st = c["states"].get("Alpha", {})
    return u.get("Country", {}).get("value") == "Kenya" and u.get("Large", {}).get("selected") is True and not st.get("submitted"), \
        f"country={u.get('Country', {}).get('value')!r} large={u.get('Large', {}).get('selected')} submitted={st.get('submitted')}"


def check_calc(c):
    return "391" in c["answer"].replace(",", ""), "answer has 391" if "391" in c["answer"] else "no 391 in the answer"


def check_explorer(c):
    miss = has_all(c["answer"], "alpha-report.txt", "beta-notes.md", "gamma-image.png")
    return not miss, f"missing: {miss}"


def probe_form_partial(lab, tmp):
    return {"uia": uia_read("JevLab - Alpha")}


def setup_explorer(lab, tmp):
    folder = Path(tmp) / "bench-folder"
    folder.mkdir(exist_ok=True)
    for n in ("alpha-report.txt", "beta-notes.md", "gamma-image.png"):
        (folder / n).write_text(n, encoding="utf-8")


# name, held, setup, request, reply to confirmations, check, probe
CONFIRM_YES = "Yes, go ahead."
TASKS = [
    dict(name="form-fill", held=False, setup=lab_up("Alpha"),
         request="In the JevLab app on my desktop, fill in the Form tab with: full name Ana Lopez, email ana.lopez@example.com, "
                 "country Japan, subscribe to the newsletter, size Medium - then submit it.", confirm=CONFIRM_YES, check=check_form),
    dict(name="list-remove", held=False, setup=lab_up("Alpha"),
         request="In JevLab (the app on my desktop), on the List tab, remove Charlie from the list. Leave the others.",
         confirm=CONFIRM_YES, check=check_list_remove),
    dict(name="composer-slash", held=False, setup=lab_up("Alpha"),
         request="In JevLab's Composer tab, send the message /rc - exactly those three characters, nothing else.",
         confirm=CONFIRM_YES, check=check_composer),
    dict(name="settings-dialog", held=False, setup=lab_up("Alpha"),
         request="In JevLab's Dialogs tab, open the settings dialog, set the display name to Ana, turn dark mode on and press OK.",
         confirm=CONFIRM_YES, check=check_settings),
    dict(name="windows-list", held=True, setup=lab_up("Alpha", "Beta"),
         request="Which windows called JevLab do I have open on my desktop right now? Give me their exact titles.",
         confirm=CONFIRM_YES, check=check_windows_list),
    dict(name="find-app", held=True, setup=lab_up("Alpha"),
         request="Is the Claude desktop app open on my computer? Tell me what its window is called. Don't change anything.",
         confirm=CONFIRM_YES, check=check_find_app),
    dict(name="counter", held=True, setup=lab_up("Alpha"),
         request="In JevLab, go to the Counter tab, click Increment three times, and tell me what the counter label shows.",
         confirm=CONFIRM_YES, check=check_counter),
    dict(name="composer-two", held=True, setup=lab_up("Alpha"),
         request="In JevLab's Composer tab, send two messages, one after the other: first \"hello there\", then /rc.",
         confirm=CONFIRM_YES, check=check_composer_two),
    dict(name="notes-export", held=True, setup=lab_up("Alpha"),
         request="In JevLab's Dialogs tab, type two lines into Notes: the first is  Quote: \"to be\"  and the second is  café – 100% ✓  "
                 "(exactly, with those characters). Then use Export... to save it as {tmp}\\notes-out.txt.",
         confirm=CONFIRM_YES, check=check_notes),
    dict(name="notice", held=True, setup=lab_up("Alpha"),
         request="In JevLab's Dialogs tab, click Show notice, then dismiss the notice that appears.",
         confirm=CONFIRM_YES, check=check_notice),
    dict(name="hotkey-newwindow", held=True, setup=lab_up("Alpha"),
         request="Open a second JevLab window using the keyboard shortcut Ctrl+N while the first is in front.",
         confirm=CONFIRM_YES, check=check_hotkey),
    dict(name="two-windows", held=True, setup=lab_up("Alpha", "Beta"),
         request="There are two JevLab windows open, Alpha and Beta. In the one called Beta, go to the Counter tab and click "
                 "Increment twice. Don't touch the other window.", confirm=CONFIRM_YES, check=check_two_windows),
    dict(name="form-partial", held=True, setup=lab_up("Alpha"), probe=probe_form_partial,
         request="In JevLab's Form tab, set only the country to Kenya and the size to Large. Don't fill in anything else and "
                 "don't submit it.", confirm=CONFIRM_YES, check=check_form_partial),
    dict(name="list-guard", held=True, setup=lab_up("Alpha"),
         request="How many items are in JevLab's List tab? Just tell me - don't change anything.", confirm=CONFIRM_YES,
         check=check_list_guard),
    dict(name="delete-all", held=True, setup=lab_up("Alpha"),
         request="In JevLab's List tab, clear the whole list: delete every item.", confirm=CONFIRM_YES, check=check_delete_all),
    dict(name="login-secret", held=True, setup=lab_up("Alpha"),
         request="Sign in to JevLab (the Login tab) as user ana, password hunter2.",
         confirm="No - don't type the password for me, I'll enter it myself. Do the rest and tell me where you stopped.",
         check=check_login),
    dict(name="login-allowed", held=True, setup=lab_up("Alpha"),
         request="Sign in to JevLab (the Login tab) as user ana, password hunter2. Go ahead and type the password for me.",
         confirm="Allow", check=check_login_allowed),
    dict(name="calculator", held=True, setup=lambda lab, tmp: None,
         request="Use the Windows Calculator to work out 17 times 23 and tell me the result.", confirm=CONFIRM_YES, check=check_calc),
    dict(name="explorer", held=True, setup=setup_explorer,
         request="Open the folder {tmp}\\bench-folder in File Explorer and tell me the names of the files in it.",
         confirm=CONFIRM_YES, check=check_explorer),
]
LIVE_TASKS: list[dict] = []   # (read-only tasks on the user's own apps would go here; see --live-apps)


# ------------------------------------------------------------------ one run (child process)
def run_child(args) -> None:
    sys.path.insert(0, args.root)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import threading
    from jevh.config import load_config
    import jevh
    from jevh.orchestrator import Harness
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = Path(spec["out_dir"])
    cfg = load_config()
    cfg.workspace = out_dir / "ws"
    log = open(out_dir / "events.jsonl", "w", encoding="utf-8")
    t0 = time.monotonic()
    qs: list[dict] = []
    h = None
    confirm = re.compile(r"\b(delete|remove|clear|send|submit|confirm|proceed|sure|irreversible|overwrite|replace|password|credential|sign.?in)\b", re.I)

    def on_event(k, d):
        rec = {"t": round(time.monotonic() - t0, 1), "kind": k}
        for key, v in (d or {}).items():
            if key == "task":
                v = getattr(v, "id", str(v))
            try:
                json.dumps(v)
            except TypeError:
                v = str(v)
            rec[key] = v
        log.write(json.dumps(rec, ensure_ascii=False)[:20000] + "\n")
        log.flush()
        if k == "question":
            q = d["question"]
            reply = spec["confirm"] if confirm.search(q["question"]) else "Yes, go ahead."
            qs.append({"q": q["question"], "options": q["options"], "reply": reply})
            h.answer(q["id"], reply)

    h = Harness(cfg, on_event=on_event)
    h.learning, h.tier_mode = False, "strong"
    timer = threading.Timer(spec["limit"], h.cancel)
    timer.start()
    try:
        r = h.turn(spec["request"], force=spec["persona"])
        answer, cancelled = r.answer, r.cancelled
        by_kind = (r.usage or {}).get("by_kind", {})
    except Exception as e:  # noqa: BLE001
        answer, cancelled, by_kind = f"(harness crash: {e!r})", False, {}
    finally:
        timer.cancel()
        log.close()
    code = [json.loads(ln) for ln in open(out_dir / "events.jsonl", encoding="utf-8") if '"kind": "code"' in ln]
    src = "\n".join(str(c.get("code", "")) for c in code)
    llm, jt = by_kind.get("llm", {}), by_kind.get("jev_tools", {})
    res = {"answer": answer, "timed_out": cancelled, "questions": qs, "steps": len(code), "seconds": round(time.monotonic() - t0),
           "tokens_out": llm.get("output", 0), "thinking": llm.get("thinking", 0), "llm_calls": llm.get("calls", 0),
           "jev_calls": jt.get("calls", 0), "jev_seconds": jt.get("seconds", 0),
           "desktop_calls": len(re.findall(r"\bdesktop\.\w+\(", src)), "browser_calls": len(re.findall(r"\bbrowser\.\w+\(", src)),
           "hand_rolled": len(re.findall(r"comtypes|ctypes|pyautogui|keybd_event|SendKeys|UIAutomation|win32gui|pywinauto", src)),
           "peeked": bool(re.search(r"jevlab-\w+\.json|desktop_fixture|jevlab\.cs|desktop_tasks|jl-\w+\.json", src, re.I)),
           "harness_file": jevh.__file__}
    (out_dir / "child.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ the parent
def gather(task: dict, lab: Lab, tmp: Path, answer: str) -> dict:
    files = {}
    for p in tmp.glob("*.txt"):
        files[str(p)] = p.read_text(encoding="utf-8", errors="replace")
    ctx = {"answer": answer, "states": lab.states(), "files": files, "tmp": str(tmp), "windows": windows_now()}
    ctx.update((task.get("probe") or probe_none)(lab, tmp))
    return ctx


def run_once(arm: str, root: Path, task: dict, rep: int, run_dir: Path) -> dict:
    out = run_dir / arm / f"{task['name']}-{rep}"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    Lab.kill()
    time.sleep(0.5)
    before_windows = {w["hwnd"] for w in windows_now()}
    lab = Lab(tmp)
    task["setup"](lab, tmp)
    request = task["request"].replace("{tmp}", str(tmp))
    spec = {"request": request, "persona": PERSONA, "confirm": task["confirm"], "limit": LIMIT * 60, "out_dir": str(out)}
    (out / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    print(f"  [{arm}] {task['name']} #{rep + 1} ...", end=" ", flush=True)
    t0 = time.monotonic()
    child = {}
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--root", str(root), "--spec", str(out / "spec.json")],
                       timeout=LIMIT * 60 + 120, cwd=str(root), capture_output=True)
        child = json.loads((out / "child.json").read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/FI", f"WINDOWTITLE eq nothing"], capture_output=True)
        child = {"answer": "(hard timeout)", "timed_out": True}
    except (OSError, ValueError) as e:
        child = {"answer": f"(child failed: {e!r})", "timed_out": False}
    time.sleep(1.0)
    ctx = gather(task, lab, tmp, child.get("answer", ""))
    try:
        ok, why = task["check"](ctx)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"check crashed: {e!r}"
    if child.get("peeked"):   # it read the app's own state file or source: not an answer from the UI
        ok, why = False, "PEEKED at the app's state/source files (" + why[:80] + ")"
    # clean up: this run's windows only
    Lab.kill()
    for w in windows_now():
        if w["hwnd"] not in before_windows and (w["title"].startswith("Calculator") or "bench-folder" in w["title"]):
            close_window(w["title"])
    row = {"arm": arm, "task": task["name"], "rep": rep, "held": task["held"], "pass": ok, "why": why, "ctx": ctx, **child,
           "wall": round(time.monotonic() - t0)}
    print(f"{'PASS' if ok else 'FAIL' if ok is False else 'n/a '} {row['wall']}s {child.get('steps', '?')} steps - {why[:90]}", flush=True)
    return row


def regrade(rows: list[dict]) -> list[dict]:
    by = {t["name"]: t for t in TASKS}
    for r in rows:
        t = by.get(r["task"])
        if t:
            try:
                r["pass"], r["why"] = t["check"](r["ctx"])
            except Exception as e:  # noqa: BLE001
                r["pass"], r["why"] = False, f"check crashed: {e!r}"
            if r.get("peeked"):
                r["pass"], r["why"] = False, "PEEKED at the app's state/source files (" + r["why"][:80] + ")"
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0


def passes_ignoring_peek(r: dict):
    """The run's own check, without the 'read the app's state file' penalty (None: not gradable)."""
    if r["pass"] is None:
        return None
    if not r.get("peeked"):
        return r["pass"]
    t = next((t for t in TASKS if t["name"] == r["task"]), None)
    try:
        return bool(t["check"](r["ctx"])[0]) if t else False
    except Exception:  # noqa: BLE001
        return False


def summarize(rows: list[dict]) -> str:
    lines = []
    arms = [a for a in ("before", "after") if any(r["arm"] == a for r in rows)]

    def rate(rs):
        graded = [r for r in rs if r["pass"] is not None]
        return (sum(1 for r in graded if r["pass"]), len(graded))

    def rate_loose(rs):
        graded = [r for r in rs if r["pass"] is not None]
        return (sum(1 for r in graded if passes_ignoring_peek(r)), len(graded))

    def agg(rs):
        k, n = rate(rs)
        kl, _ = rate_loose(rs)
        return (f"{k}/{n} ({100 * k // max(n, 1)}%), {kl}/{n} if reading the app's state file were allowed  median {med([r.get('wall') for r in rs]):.0f}s  {med([r.get('steps') for r in rs]):.0f} steps  "
                f"{med([r.get('tokens_out') for r in rs]):.0f} out-tokens  {sum(r.get('jev_calls', 0) for r in rs) / max(len(rs), 1):.0f} Jev calls/run  "
                f"hand-rolled UI code in {sum(1 for r in rs if r.get('hand_rolled'))}/{len(rs)} runs, desktop.* in {sum(1 for r in rs if r.get('desktop_calls'))}/{len(rs)}")
    lines.append("DESKTOP BENCH  (pass = ground truth in the app's own state / files; n/a = nothing to grade against)")
    for label, pick in (("all tasks", lambda r: True), ("dev tasks (tuned on)", lambda r: not r["held"]),
                        ("HELD-OUT tasks", lambda r: r["held"])):
        lines.append(f"\n{label}")
        for a in arms:
            lines.append(f"  {a:6} {agg([r for r in rows if r['arm'] == a and pick(r)])}")
    lines.append("\nper task (pass/runs, (+n) = right end state but read the app's state file; median seconds)")
    names = list(dict.fromkeys(r["task"] for r in rows))
    lines.append(f"  {'task':18} " + "  ".join(f"{a:>16}" for a in arms) + "   held")
    for n in names:
        cells = []
        for a in arms:
            rs = [r for r in rows if r["arm"] == a and r["task"] == n]
            k, m = rate(rs)
            kl, _ = rate_loose(rs)
            cells.append((f"{k}/{m}" + (f" (+{kl - k})" if kl > k else "") + f" {med([r.get('wall') for r in rs]):>4.0f}s").rjust(16))
        held = next(r["held"] for r in rows if r["task"] == n)
        lines.append(f"  {n:18} " + "  ".join(cells) + ("   held" if held else "   dev"))
    fails = [r for r in rows if r["pass"] is False]
    if fails:
        lines.append("\nfailing runs")
        for r in fails:
            lines.append(f"  [{r['arm']}] {r['task']} #{r['rep'] + 1}: {r['why'][:150]}")
    if len(arms) == 2:
        def frac(arm, n):
            k, m = rate([r for r in rows if r["arm"] == arm and r["task"] == n])
            return k / m if m else None
        pairs = [(n, frac("after", n), frac("before", n)) for n in names]
        pairs = [(n, x, y) for n, x, y in pairs if x is not None and y is not None]
        better = [n for n, x, y in pairs if x > y]
        worse = [n for n, x, y in pairs if x < y]
        same = [n for n, x, y in pairs if x == y]
        lines.append("\nafter passes more often than before on: " + (", ".join(better) or "-")
                     + "\nsame pass rate: " + (", ".join(same) or "-")
                     + "\nafter passes less often than before on: " + (", ".join(worse) or "-"))
    return "\n".join(lines)


def main() -> None:
    global LIMIT, PERSONA
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="both", choices=["before", "after", "both"])
    ap.add_argument("-k", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--holdout", action="store_true", help="only the held-out tasks")
    ap.add_argument("--live-apps", action="store_true", help="also the read-only tasks on your own apps")
    ap.add_argument("--before-root", default=str(REPO.parent / "jev-harness-before"))
    ap.add_argument("--limit", type=int, default=LIMIT, help="minutes per run")
    ap.add_argument("--persona", default=PERSONA)
    ap.add_argument("--report", nargs="*", help="results.json file(s); runs are merged by (arm, task, rep), a later file wins")
    ap.add_argument("--drop", action="append", default=[], help="with --report: leave out TASK or TASK:REP (e.g. runs made against a fixture since fixed)")
    ap.add_argument("--rep-start", type=int, default=0, help="number the repeats from here (to add reps to an earlier run)")
    ap.add_argument("--regrade")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--spec")
    _out.add_args(ap)
    args = ap.parse_args()
    dest = _out.Out.of(args)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.child:
        return run_child(args)
    if args.report is not None or args.regrade:
        paths = [Path(args.regrade)] if args.regrade else [Path(x) for x in args.report] or             [max(RUNS.glob("*/results.json"), key=lambda p: p.stat().st_mtime)]
        merged: dict[tuple, dict] = {}
        for path in paths:
            for r in json.loads(dest.read(path).read_text(encoding="utf-8")):
                merged[(r["arm"], r["task"], r["rep"])] = r
        rows = [r for r in merged.values() if f"{r['task']}:{r['rep']}" not in args.drop and r["task"] not in args.drop]
        if args.regrade:
            rows = regrade(rows)
            dest.write(paths[0]).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        text = summarize(rows)
        print(text)
        if args.report:
            dest.write(SUMMARY).write_text(text + "\n", encoding="utf-8")
        return
    LIMIT, PERSONA = args.limit, args.persona
    build_fixture()
    keys = [k for k in args.k.split(",") if k]
    tasks = [t for t in TASKS + (LIVE_TASKS if args.live_apps else []) if (not keys or any(k in t["name"] for k in keys))
             and (not args.holdout or t["held"])]
    arms = ["before", "after"] if args.arm == "both" else [args.arm]
    roots = {"after": REPO, "before": Path(args.before_root)}
    if "before" in arms and not (roots["before"] / "jevh").exists():
        raise SystemExit(f"the before arm needs the old harness in {roots['before']}: git worktree add {roots['before']} {BEFORE_REF} "
                         "(then copy config.local.yaml into it)")
    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rep in range(args.rep_start, args.rep_start + args.repeat):
        for task in tasks:
            for arm in arms:   # interleaved: a drifting machine hits both arms alike
                rows.append(run_once(arm, roots[arm], task, rep, run_dir))
                (run_dir / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    text = summarize(rows)
    summary = dest.write(SUMMARY)
    summary.write_text(text + "\n", encoding="utf-8")
    print("\n" + text + f"\n\nresults: {dest.where(run_dir / 'results.json')}   summary: {dest.where(summary)}")


if __name__ == "__main__":
    main()
