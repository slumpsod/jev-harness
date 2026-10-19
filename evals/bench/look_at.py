"""look_at(): the worker looks at a picture in a file with its own model (issue #2), measured with the live models.

The bug (issue #2, an outside user; and session 20261012-153059-7780 here: "what software are they using in this video?"
with a screenshot): asked about scanned PDFs and photos, the worker found no way to show a file to its model, decided it
had no vision and went for OCR (pytesseract, a Tesseract installer, winget, rapidocr), with poor results.

Two parts:

  turns    Full harness turns (daily_assistant, the live worker model and Jev): a question about an attached file, sent
           the way the dashboard sends it (the message, then the note naming the file; one case per split names the
           path in the text instead). Old arm: BEFORE_REF (no look_at) vs new: this checkout. Each arm runs in its own
           throwaway venv with the packages that arm's pyproject installs (the old one has no PyMuPDF), plus a
           sitecustomize that BLOCKS installs (pip/uv/conda/winget/choco/msiexec, running a downloaded installer) and
           opening a file in an app on the screen (a live old-arm worker opened the receipt in Photos to look at it with
           the desktop tool), and logs each attempt: no worker can install software or put windows on this machine's
           screen. For the same reason the persona runs without its desktop and devices tools, in both arms.
           fire cases: a scanned form (PNG), an image-only PDF invoice (2 pages, the total on page 2), a phone photo of
           a receipt (stored sideways with an EXIF tag), a screenshot of a video page (the real case above).
           lookalikes, which must NOT use vision: the same invoice as a text PDF, a CSV.
           held out (--holdout): a photo of a handwritten sticky note (path in the text), a 3-page scanned lease (the
           answer on page 3), a text PDF named like a scan (scan_0042.pdf, path in the text), a .txt.
  direct   look_at() itself on every picture file, no worker: the model's reading of each, checked against what the
           file holds (new arm only; the old one has no look_at).

    python evals/bench/look_at.py                     # turns, both arms, 3 repeats, main split
    python evals/bench/look_at.py --holdout           # the held-out cases
    python evals/bench/look_at.py --part direct       # look_at() alone, 3 repeats
    python evals/bench/look_at.py --arm new -k receipt --repeat 1
    python evals/bench/look_at.py --offline           # grade look_at_raw.json again (no models)

Checks per turn: the answer holds the known values (correct); installs attempted (blocked by the venv, plus pip/winget
calls found in the code it ran); OCR tried (an OCR package or Tesseract in its code); vision used (look_at or a tool's
look() in its code). A fire case passes when correct with no install attempted; a lookalike when correct without vision
or OCR. Reported per arm with rates, time, steps and model tokens.

Needs: the worker model and Jev your config names (designed on Qwen3.8 27B with its vision projector on llama-server,
and the local semif Jev), Pillow and PyMuPDF in the Python running this (it makes the files), and network access once
to build the two venvs (%TEMP%/jevh-lookat-venv-old, -new). Raw: look_at_raw.json; summary: look_at_summary.txt
(look_at_holdout_summary.txt, look_at_direct_summary.txt).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

BEFORE_REF = "15cc4bd"          # origin/main before look_at
PERSONA = "daily_assistant"
LIMIT = 6 * 60                  # seconds per turn
RAW = HERE / "look_at_raw.json"
FILES = Path(tempfile.gettempdir()) / "jevh-lookat-files"
PACKAGES = {"old": ["pyyaml", "rich", "pillow", "numpy", "comtypes", "playwright==1.62.0"],
            "new": ["pyyaml", "rich", "pillow", "numpy", "comtypes", "playwright==1.62.0", "pymupdf"]}

CASES = [
    {"name": "scanned_form", "kind": "fire", "file": "intake_form_scan.png",
     "text": "I scanned my new-patient form. What insurance policy number and date of birth did I write on it?",
     "expect": [r"HX\W?44721\W?B", r"03/14/1987|march 14,? 1987|14 march,? 1987|1987-03-14"]},
    {"name": "image_pdf_invoice", "kind": "fire", "file": "invoice_INV-scan.pdf",
     "text": "This is the plumbing supply invoice, scanned to PDF. What's the invoice number and the total due?",
     "expect": [r"INV\W?20931", r"1,?284\.50"]},
    {"name": "receipt_photo", "kind": "fire", "file": "IMG_4471.jpg",
     "text": "photo of my receipt from this morning - which shop was it and how much did I pay in total?",
     "expect": [r"corner crow", r"48\.37"]},
    {"name": "screenshot_video", "kind": "fire", "file": "screen.png",   # real: session 20261012-153059-7780
     "text": "can you tell me what software they are using in this video im watching? here's a screenshot",
     "expect": [r"blender"]},
    {"name": "text_pdf_invoice", "kind": "lookalike", "file": "invoice_INV-20931.pdf",
     "text": "Here's the invoice PDF from the plumbing supplier. What's the total due?",
     "expect": [r"1,?284\.50"]},
    {"name": "csv_expenses", "kind": "lookalike", "file": "october_expenses.csv",
     "text": "How much did I spend in total in this expenses file, and which category was the biggest?",
     "expect": [r"263\.82", r"groceries"]},
    # held out
    {"name": "sticky_note", "kind": "fire", "holdout": True, "file": "PXL_20261016_sticky.jpg", "inline": True,
     "text": "I took a photo of a sticky note, it's at {path} - when is the appointment and what number do I call if I'm late?",
     "expect": [r"3:45", r"555\W?0182"]},
    {"name": "lease_scan_page3", "kind": "fire", "holdout": True, "file": "lease_signed_scan.pdf",
     "text": "When does my lease end and what's the monthly rent? This is the signed copy I scanned.",
     "expect": [r"july 31,? 2027|31 july,? 2027|2027-07-31|07/31/2027", r"1,?950"]},
    {"name": "text_pdf_named_scan", "kind": "lookalike", "holdout": True, "file": "scan_0042.pdf", "inline": True,
     "text": "What's the claim reference and the approved amount in {path}?",
     "expect": [r"CLM\W?7730\W?QX", r"612\.40"]},
    {"name": "notes_txt", "kind": "lookalike", "holdout": True, "file": "meeting_notes.txt",
     "text": "Who owns the follow-up from these meeting notes, and when was the meeting?",
     "expect": [r"priya", r"nov(ember)?\.? 4|11/4|4 nov"]},
]
# what look_at() alone is asked in the direct part (every picture file; the expectations as above)
DIRECT = [("intake_form_scan.png", "What are the insurance policy number and the date of birth?", CASES[0]["expect"]),
          ("invoice_INV-scan.pdf", "What are the invoice number and the total due?", CASES[1]["expect"]),
          ("IMG_4471.jpg", "Which shop is this receipt from and what is the total?", CASES[2]["expect"]),
          ("screen.png", "Which software is shown in the video?", CASES[3]["expect"]),
          ("PXL_20261016_sticky.jpg", "When is the appointment and what number to call?", CASES[6]["expect"]),
          ("lease_signed_scan.pdf", "When does the lease end and what is the monthly rent?", CASES[7]["expect"])]

# --------------------------------------------------------------- what the worker's code did
# OCR software named in code it ran (importing it, installing it, downloading Tesseract)
OCR = re.compile(r"pytesseract|easyocr|rapidocr|paddleocr|ocrmypdf|keras_ocr|\bdoctr\b|tesseract|UB-Mannheim", re.I)
# code that RUNS an install: pip (shell, subprocess, its API), winget/choco/msiexec, a downloaded installer started
INSTALL = re.compile(r"(subprocess\.\w+|os\.system|os\.popen|start_job|os\.startfile|\bsh)\s*\([^)]*"
                     r"(\bpip3?\b|\bwinget\b|\bchoco\b|\bmsiexec\b|\bconda\b|\buv\b|\.exe|\.msi)|"
                     r"pip\.main|pip\._internal|ensurepip|^\s*!\s*pip", re.I | re.M)
VISION = re.compile(r"\blook_at\s*\(|\.look\s*\(")

SITECUSTOMIZE = r'''# jevh look_at bench: in a run with JEVH_BENCH_BLOCKLOG set, installing software is refused and logged.
import os, re, sys
_LOG = os.environ.get("JEVH_BENCH_BLOCKLOG")
if _LOG:
    _RX = re.compile(r"\bpip3?(\.exe)?\W+(install|download)\b|-m\W+pip\b|\bwinget\b|\bchoco(latey)?\b|\bscoop\b|"
                     r"\bmsiexec\b|\bconda\b.*\binstall\b|\buv\b\W+(pip\W+)?(add|install)\b|\bnpm\b.*\binstall\b|"
                     r"Add-AppxPackage|Install-(Package|Module)|\.msi\b|tesseract[^\s'\"]*\.exe|(setup|install)[^\s'\"\\/]*\.exe",
                     re.I)

    # opening a file in an app on the real screen (a live old-arm worker opened the receipt in Photos to look at it with
    # the desktop tool): refused too, so a bench never puts windows on the screen of the person using this machine
    _OPEN = re.compile(r"^\s*(cmd(\.exe)?\s+/c\s+)?start\b|\bexplorer(\.exe)?\s|ms-photos:|\bmspaint\b|Photos\.exe|"
                       r"Invoke-Item|\bii\s|Start-Process", re.I)

    def _block(what, kind="install"):
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(kind + ": " + str(what)[:400].replace("\n", " ") + "\n")
        raise PermissionError("Installing software is not allowed on this computer." if kind == "install" else
                              "Opening files in apps on the screen is not allowed here.")

    def _check(args):
        s = args if isinstance(args, str) else " ".join(map(str, args)) if isinstance(args, (list, tuple)) else str(args)
        if _RX.search(s):
            _block(s)
        if _OPEN.search(s):
            _block(s, "open")

    import subprocess
    _init = subprocess.Popen.__init__

    def _popen(self, args, *a, **k):
        _check(args)
        return _init(self, args, *a, **k)
    subprocess.Popen.__init__ = _popen
    _system = os.system
    os.system = lambda cmd: (_check(cmd), _system(cmd))[1]
    if hasattr(os, "startfile"):

        def _startfile(path, *a, **k):
            if re.search(r"\.(exe|msi|msix|appx)$", str(path), re.I):
                _block(f"startfile {path}")
            _block(f"startfile {path}", "open")
        os.startfile = _startfile

    class _NoPip:
        def find_spec(self, name, path=None, target=None):
            if name == "pip" or name.startswith("pip."):
                _block(f"import {name}")
            return None
    sys.meta_path.insert(0, _NoPip())
'''


def venv(arm: str) -> str:
    """The arm's throwaway Python: what its pyproject installs, and installs blocked (made once, then reused)."""
    root = Path(tempfile.gettempdir()) / f"jevh-lookat-venv-{arm}"
    exe = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(root)], check=True)
        subprocess.run([str(exe), "-m", "pip", "install", "-q", *PACKAGES[arm]], check=True)
    site = subprocess.run([str(exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    Path(site, "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return str(exe)


def export_before(ref: str) -> str:
    """The harness at `ref` in a temporary folder, with this checkout's config.local.yaml."""
    dest = Path(tempfile.gettempdir()) / f"jevh-lookat-before-{ref}"
    if not (dest / "jevh").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        tar = dest / "src.tar"
        subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(tar), ref, "jevh", "config.yaml", "personas", "skills"],
                       check=True)
        import tarfile
        with tarfile.open(tar) as t:
            t.extractall(dest, filter="data")
        tar.unlink()
    if (REPO / "config.local.yaml").is_file():
        shutil.copy(REPO / "config.local.yaml", dest / "config.local.yaml")
    return str(dest)


def request_text(case: dict, files: dict[str, str]) -> str:
    path = files[case["file"]]
    if case.get("inline"):
        return case["text"].format(path=path)
    sys.path.insert(0, str(REPO))
    from jevh.web.uploads import with_files   # (as the dashboard sends a message with a file)
    return with_files(case["text"], [Path(path)])


# --------------------------------------------------------------- one turn / one look (a child process)
def child(case: dict, root: str, rep: int, text: str) -> dict:
    sys.path.insert(0, root)
    from jevh.config import load_config
    from jevh.orchestrator import Harness
    tmp = Path(tempfile.mkdtemp(prefix="jevh-lookat-"))
    cfg = load_config(Path(root) / "config.yaml")
    cfg.workspace = tmp / "ws"
    cfg.raw = {**cfg.raw, "mcp_servers": {}}   # (no MCP servers started: not what this measures)
    out = {"task": case["name"], "rep": rep, "root": root, "python": sys.executable, "codes": [], "outputs": [],
           "questions": [], "errors": []}
    t0 = time.monotonic()
    box: dict = {}

    def on_event(k, d):
        if k == "code":
            out["codes"].append(str(d.get("code", ""))[:4000])
        elif k == "output":
            err = d.get("error")
            out["outputs"].append((str(d.get("output", "")) + (f"\n{err}" if isinstance(err, str) and err else ""))[:2000])
        elif k == "question":
            q = d["question"]
            reply = "I don't know - do what you can with the file I gave you."
            out["questions"].append({"q": q["question"][:400], "reply": reply})
            box["h"].answer(q["id"], reply)
    try:
        h = Harness(cfg, on_event=on_event)
        box["h"] = h
        h.learning = False
        # no desktop tool (it would act on the real screen of the person using this machine) and no other computers
        from dataclasses import replace
        p = h.personas[PERSONA]
        h.personas[PERSONA] = replace(p, tools=[t for t in p.tools if t not in ("desktop", "devices")])
        timer = threading.Timer(LIMIT, h.cancel)
        timer.start()
        try:
            r = h.turn(text, force=PERSONA)
            out.update(answer=r.answer, seconds=round(time.monotonic() - t0, 1), cancelled=bool(r.cancelled),
                       usage=(r.usage or {}).get("total"))
        finally:
            timer.cancel()
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    finally:
        log = os.environ.get("JEVH_BENCH_BLOCKLOG")
        out["blocked"] = Path(log).read_text(encoding="utf-8").splitlines() if log and Path(log).is_file() else []
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def child_direct(name: str, question: str, root: str, rep: int, path: str) -> dict:
    """look_at() in a REPL whose `look` handler is the worker model, as in a turn (run_task's handlers)."""
    sys.path.insert(0, root)
    from jevh.config import load_config
    from jevh.llm import see_image
    from jevh.orchestrator import Harness
    from jevh.repl.client import Repl
    tmp = Path(tempfile.mkdtemp(prefix="jevh-lookat-direct-"))
    cfg = load_config(Path(root) / "config.yaml")
    cfg.workspace = tmp / "ws"
    cfg.raw = {**cfg.raw, "mcp_servers": {}}
    h = Harness(cfg)
    llm = h._llm(h.personas[PERSONA], "strong")
    calls = []

    def look(p, q):
        t = time.monotonic()
        a = see_image(llm, p, q)
        calls.append(round(time.monotonic() - t, 1))
        return a
    out = {"task": name, "rep": rep, "root": root}
    t0 = time.monotonic()
    with Repl(tmp / "w", tmp / "a", {"look": look}, timeout=600, output_cap=6000) as r:
        res = r.run(f"print(look_at({path!r}, {question!r}))")
    out.update(answer=res.output, error=res.error, seconds=round(time.monotonic() - t0, 1), looks=calls,
               usage=h.usage.summary(h.session).get("total"))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# --------------------------------------------------------------- checks
def grade(case: dict, r: dict) -> dict:
    code = "\n".join(r.get("codes", []))
    answer = r.get("answer") or ""
    correct = all(re.search(x, answer, re.I) for x in case["expect"])
    blocked = r.get("blocked", [])
    installs = sum(not b.startswith("open:") for b in blocked) + len(INSTALL.findall(code))
    opened = sum(b.startswith("open:") for b in blocked)
    ocr = sum(bool(OCR.search(c)) for c in r.get("codes", []))
    # a model look: look_at() or a tool's look() - not a look_at() that read a PDF's text layer and looked at nothing
    outs = r.get("outputs", [])
    vision = any(VISION.search(c) and "read from the PDF's text layer" not in (outs[i] if i < len(outs) else "")
                 for i, c in enumerate(r.get("codes", [])))
    if case["kind"] == "fire":
        checks = {"correct": correct, "no_install": installs == 0}
    else:
        checks = {"correct": correct, "no_vision": not vision, "no_ocr": ocr == 0, "no_install": installs == 0}
    return {"checks": checks, "pass": all(checks.values()), "correct": correct, "installs": installs, "ocr_steps": ocr,
            "vision": vision, "opened": opened}


def summarize(results: list[dict], title: str) -> str:
    lines = [title, ""]
    for arm in sorted({r["arm"] for r in results}, key=lambda a: a != "old"):
        rs = [r for r in results if r["arm"] == arm]
        lines.append(f"== {arm}")
        tot = {"pass": 0, "n": 0, "secs": 0.0, "steps": 0, "tok": 0, "calls": 0}
        fails = []
        for c in CASES:
            mine = [r for r in rs if r["task"] == c["name"]]
            if not mine:
                continue
            g = [grade(c, r) for r in mine]
            n, ok = len(g), sum(x["pass"] for x in g)
            secs = [r.get("seconds") or 0 for r in mine]
            steps = [len(r.get("codes", [])) for r in mine]
            toks = [((r.get("usage") or {}).get("input") or 0) + ((r.get("usage") or {}).get("output") or 0) for r in mine]
            calls = [(r.get("usage") or {}).get("calls") or 0 for r in mine]
            tot["pass"] += ok
            tot["n"] += n
            tot["secs"] += sum(secs)
            tot["steps"] += sum(steps)
            tot["tok"] += sum(toks)
            tot["calls"] += sum(calls)
            lines.append(f"  {c['name']:22s} {c['kind']:9s} {ok}/{n} pass" + (" (held out)" if c.get("holdout") else "")
                         + f"  correct {sum(x['correct'] for x in g)}/{n}, vision used {sum(x['vision'] for x in g)}/{n}, "
                           f"OCR tried {sum(x['ocr_steps'] > 0 for x in g)}/{n} ({sum(x['ocr_steps'] for x in g)} steps), "
                           f"installs attempted {sum(x['installs'] for x in g)}, opened in an app {sum(x['opened'] for x in g)}")
            lines.append(f"      avg {sum(secs) / n:.0f}s, {sum(steps) / n:.1f} code steps, {sum(toks) / n / 1000:.1f}k tokens, "
                         f"{sum(calls) / n:.1f} model calls")
            for r, x in zip(mine, g):
                if not x["pass"]:
                    why = [k for k, v in x["checks"].items() if not v]
                    fails.append(f"{c['name']} rep {r['rep']}: {', '.join(why)} - {str(r.get('answer') or r.get('errors'))[:140]!r}")
                for e in r.get("errors", [])[:1]:
                    lines.append(f"      error (rep {r['rep']}): {e}")
        n = max(tot["n"], 1)
        lines.append(f"  total: {tot['pass']}/{tot['n']} pass; per turn avg {tot['secs'] / n:.0f}s, {tot['steps'] / n:.1f} "
                     f"code steps, {tot['tok'] / n / 1000:.1f}k tokens, {tot['calls'] / n:.1f} model calls")
        lines += [f"    FAIL {f}" for f in fails]
        lines.append("")
    return "\n".join(lines)


def summarize_direct(results: list[dict], title: str) -> str:
    lines = [title, ""]
    ok = n = 0
    for name, _, expect in DIRECT:
        mine = [r for r in results if r["task"] == name]
        if not mine:
            continue
        good = [all(re.search(x, r.get("answer") or "", re.I) for x in expect) and not r.get("error") for r in mine]
        looks = [s for r in mine for s in r.get("looks", [])]
        toks = [((r.get("usage") or {}).get("input") or 0) + ((r.get("usage") or {}).get("output") or 0) for r in mine]
        ok += sum(good)
        n += len(good)
        lines.append(f"  {name:26s} {sum(good)}/{len(good)} read right; {len(looks) / len(mine):.0f} look(s), "
                     f"avg {sum(looks) / max(len(looks), 1):.1f}s a look, {sum(toks) / len(mine) / 1000:.1f}k tokens")
        for r, g in zip(mine, good):
            if not g:
                lines.append(f"      MISS rep {r['rep']}: {str(r.get('answer'))[:200]!r}")
    lines.append(f"  total: {ok}/{n}")
    return "\n".join(lines)


def run_child(args: list[str], py: str, env: dict) -> dict:
    """One child; past its time (the turn's own limit did not stop it) its whole process tree is ended."""
    p = subprocess.Popen([py, __file__, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO),
                         env=env, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    try:
        out, err = p.communicate(timeout=LIMIT + 180)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        out, err = p.communicate()
        return {"errors": [f"child did not finish in {LIMIT + 180}s and was ended"], "answer": "", "seconds": LIMIT + 180}
    line = next((x for x in out.splitlines() if x.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"errors": [f"child failed: {err.strip()[-400:]}"]}


def save(raw: Path, old: list[dict], results: list[dict]) -> None:
    """The raw results so far (after every turn: a run cut short keeps what it did), replacing the same arm/case/rep."""
    keep = {(r["arm"], r["task"], r["rep"]) for r in results}
    raw.write_text(json.dumps([r for r in old if (r.get("arm"), r["task"], r["rep"]) not in keep] + results, indent=1,
                              default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="turns", choices=["turns", "direct"])
    ap.add_argument("--arm", default="both", choices=["old", "new", "both"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("-k", default="")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--child")
    ap.add_argument("--direct")
    ap.add_argument("--root")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--text")
    ap.add_argument("--path")
    a = ap.parse_args()
    if a.child:
        case = next(c for c in CASES if c["name"] == a.child)
        print("RESULT " + json.dumps(child(case, a.root, a.rep, a.text), default=str))
        return
    if a.direct:
        q = next(q for f, q, _ in DIRECT if f == a.direct)
        print("RESULT " + json.dumps(child_direct(a.direct, q, a.root, a.rep, a.path), default=str))
        return
    split = "holdout" if a.holdout else "main"
    raw = RAW if a.part == "turns" else HERE / "look_at_direct_raw.json"
    summary = HERE / ("look_at_direct_summary.txt" if a.part == "direct" else f"look_at{'_holdout' if a.holdout else ''}_summary.txt")
    if a.offline:
        old = json.loads(raw.read_text(encoding="utf-8"))
        if a.part == "direct":
            print(summarize_direct(old, f"look_at direct (offline re-grade of {raw.name})"))
        else:
            names = {c["name"] for c in CASES if bool(c.get("holdout")) == a.holdout}
            print(summarize([r for r in old if r["task"] in names], f"look_at turns, {split} (offline re-grade of {raw.name})"))
        return
    sys.path.insert(0, str(HERE))
    from look_at_files import make
    files = make(FILES)
    old = json.loads(raw.read_text(encoding="utf-8")) if raw.is_file() else []
    results = []
    if a.part == "direct":
        py = venv("new")
        for rep in range(a.repeat):
            for name, _, _ in DIRECT:
                if a.k and not any(k in name for k in a.k.split(",")):
                    continue
                print(f"direct {name:26s} rep {rep} ...", flush=True)
                r = run_child(["--direct", name, "--root", str(REPO), "--rep", str(rep), "--path", files[name]], py,
                              dict(os.environ))
                r.update(task=name, rep=rep, arm="new")
                results.append(r)
                save(raw, old, results)
                print(f"       -> {r.get('seconds')}s {str(r.get('answer'))[:200]!r}", flush=True)
        text = summarize_direct(results, f"look_at direct: {len(results)} looks, {time.strftime('%Y-%m-%d %H:%M')}")
    else:
        cases = [c for c in CASES if bool(c.get("holdout")) == a.holdout]
        if a.k:
            cases = [c for c in CASES if any(k in c["name"] for k in a.k.split(","))]
        arms = [("old", export_before(BEFORE_REF)), ("new", str(REPO))]
        arms = [x for x in arms if a.arm in ("both", x[0])]
        for rep in range(a.repeat):
            for c in cases:
                for arm, root in arms:
                    py = venv(arm)
                    log = Path(tempfile.gettempdir()) / f"jevh-lookat-blocked-{arm}-{c['name']}-{rep}.log"
                    log.unlink(missing_ok=True)
                    env = dict(os.environ, JEVH_BENCH_BLOCKLOG=str(log))
                    env["PATH"] = str(Path(py).parent) + os.pathsep + env.get("PATH", "")   # (its pip first on PATH)
                    print(f"{arm:4s} {c['name']:22s} rep {rep} ...", flush=True)
                    r = run_child(["--child", c["name"], "--root", root, "--rep", str(rep), "--text", request_text(c, files)],
                                  py, env)
                    r.update(task=c["name"], rep=rep, arm=arm)
                    results.append(r)
                    save(raw, old, results)
                    g = grade(c, r)
                    print(f"       -> {'pass' if g['pass'] else 'FAIL'} {[k for k, v in g['checks'].items() if not v] or ''} "
                          f"vision={g['vision']} ocr={g['ocr_steps']} installs={g['installs']} ({r.get('seconds')}s) "
                          f"{str(r.get('answer', ''))[:160]!r}", flush=True)
        text = summarize(results, f"look_at turns, {split}: {len(results)} turns, {time.strftime('%Y-%m-%d %H:%M')}")
    save(raw, old, results)
    summary.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
