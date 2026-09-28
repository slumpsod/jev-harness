"""Live eval: is a long-running worker making progress, or going in circles?

The progress check runs every few steps over the recent work log. A false "stuck" interrupts a
long task that was fine (costly: quality is king here), so it must be rare; a missed "stuck"
only costs steps until the next check or the step limit.

    python evals/progress_cases.py
    python evals/progress_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402

RAW = Path(__file__).with_name("progress_cases_raw.json")


def log(*steps):
    return "\n".join(f"CODE:\n{c}\nOUTPUT: {o}" for c, o in steps)


CASES = [  # (request, recent work log, progressing?)
    ("Build a workout tracker web app with charts and CSV export.", log(
        ("html = build_form(); save_artifact('index.html', html)", "artifacts/index.html"),
        ("add_chart_js('index.html')", "chart added: 1 canvas"),
        ("add_csv_export('index.html')", "export button wired"),
        ("browser.goto('file:///.../index.html'); browser.do('log a set of bench 100x5')", "done"),
        ("print(browser.read('what does the chart show?'))", "Bench e1RM 116.7 on 2026-09-28")), True),
    ("Fix the failing tests in my project.", log(
        ("print(project.sh('pytest -q')['out'])", "3 failed, 12 passed"),
        ("print(project.read('app/db.py', 1, 60))", "... def get(id): ..."),
        ("project.edit('app/db.py', 'def get(id):', 'def get(record_id):')", "edited app/db.py"),
        ("print(project.sh('pytest -q')['out'])", "1 failed, 14 passed"),
        ("print(project.read('app/jobs.py', 20, 60))", "... retry(3) ...")), True),
    ("Fix the failing tests in my project.", log(
        ("print(project.sh('pytest -q')['out'])", "1 failed: test_retry - AssertionError"),
        ("project.edit('jobs.py', 'retry(3)', 'retry(4)')", "edited jobs.py"),
        ("print(project.sh('pytest -q')['out'])", "1 failed: test_retry - AssertionError"),
        ("project.edit('jobs.py', 'retry(4)', 'retry(3)')", "edited jobs.py"),
        ("print(project.sh('pytest -q')['out'])", "1 failed: test_retry - AssertionError"),
        ("project.edit('jobs.py', 'retry(3)', 'retry(5)')", "edited jobs.py")), False),
    ("Make a hero image for the site.", log(
        ("paths = comfy.run(wf)", "RuntimeError: ComfyUI 400: value_not_in_list ckpt_name"),
        ("comfy.set(wf, {'4.ckpt_name': 'qwen.safetensors'}); comfy.run(wf)", "RuntimeError: ComfyUI 400: value_not_in_list"),
        ("comfy.set(wf, {'4.ckpt_name': 'Qwen.safetensors'}); comfy.run(wf)", "RuntimeError: ComfyUI 400: value_not_in_list"),
        ("comfy.set(wf, {'4.ckpt_name': 'qwen_image.safetensors'}); comfy.run(wf)", "RuntimeError: ComfyUI 400: value_not_in_list")),
     False),
    ("Make a hero image for the site.", log(
        ("paths = comfy.run(wf)", "RuntimeError: ComfyUI 400: value_not_in_list ckpt_name"),
        ("print([m for m in comfy.models('checkpoints') if 'qwen' in m.lower()])", "['Qwen/qwen_image_2.1_fp8.safetensors']"),
        ("comfy.set(wf, {'4.ckpt_name': 'Qwen/qwen_image_2.1_fp8.safetensors'}); print(comfy.run(wf))",
         "['artifacts/ComfyUI_00012_.png']")), True),
    ("Research the best rest times for hypertrophy and summarize the evidence.", log(
        ("r = web.ask('rest between sets hypertrophy evidence')", "2 passages"),
        ("r2 = web.ask('Schoenfeld 2016 rest interval study')", "2 passages"),
        ("r3 = web.ask('meta-analysis rest interval muscle growth 2024')", "2 passages"),
        ("notes = llm_query('compare these findings', r['passages'] + r2['passages'] + r3['passages'])",
         "Longer rests (2-3 min) favour hypertrophy for compounds...")), True),
    ("Research the best rest times for hypertrophy and summarize the evidence.", log(
        ("r = web.ask('rest between sets hypertrophy')", "2 passages"),
        ("r = web.ask('rest between sets for hypertrophy')", "2 passages"),
        ("r = web.ask('how long to rest between sets hypertrophy')", "2 passages"),
        ("r = web.ask('rest between sets hypertrophy best')", "2 passages"),
        ("r = web.ask('rest time between sets for muscle hypertrophy')", "2 passages")), False),
    ("Scrape the product names and prices from the shop's catalogue (5 pages).", log(
        ("browser.goto(url); print(browser.do('open the catalogue')['status'])", "done"),
        ("rows += extract(browser.read())", "12 rows"),
        ("print(browser.do('go to the next page')['status']); rows += extract(browser.read())", "done; 24 rows"),
        ("print(browser.do('go to the next page')['status']); rows += extract(browser.read())", "done; 36 rows"),
        ("print(browser.do('go to the next page')['status']); rows += extract(browser.read())", "done; 48 rows")),
     True),
    ("Scrape the product names and prices from the shop's catalogue (5 pages).", log(
        ("print(browser.do('go to the next page'))", "{'status': 'stuck', 'note': 'no element fits'}"),
        ("print(browser.do('open page 2'))", "{'status': 'stuck'}"),
        ("print(browser.do('click next'))", "{'status': 'stuck'}"),
        ("print(browser.do('go to page two of the catalogue'))", "{'status': 'stuck'}"),
        ("print(browser.do('next page please'))", "{'status': 'ambiguous'}")), False),
    ("Refactor the payroll module into smaller functions and keep the tests green.", log(
        ("print(project.read('payroll.py'))", "... 400 lines ..."),
        ("project.write('payroll/calc.py', calc_src)", "wrote payroll/calc.py"),
        ("project.write('payroll/export.py', export_src)", "wrote payroll/export.py"),
        ("project.edit('payroll.py', old_block, 'from .calc import gross, net')", "edited payroll.py"),
        ("print(project.sh('pytest -q')['out'])", "18 passed")), True),
    ("Refactor the payroll module into smaller functions and keep the tests green.", log(
        ("print(project.read('payroll.py', 1, 400))", "... 400 lines ..."),
        ("print(project.read('payroll.py', 1, 400))", "... 400 lines ..."),
        ("print(project.grep('def ', 'payroll.py'))", "22 matches"),
        ("print(project.read('payroll.py', 1, 400))", "... 400 lines ..."),
        ("print(project.tree())", "payroll.py tests/")), False),
    ("Deploy the site to Cloudflare Pages.", log(
        ("print(project.sh('npm run build')['out'])", "Error: VITE_API_URL is not defined"),
        ("project.write('.env.production', 'VITE_API_URL=https://api.example.com')", "wrote .env.production"),
        ("print(project.sh('npm run build')['out'])", "built in 4.2s"),
        ("print(project.sh('npx wrangler pages deploy dist')['out'])", "Deployment complete: https://shop.pages.dev")),
     True),
    ("Train a small classifier on the dataset and report accuracy.", log(
        ("job = start_job('python train.py --epochs 20')", "job j1 started"),
        ("print(job_status(job))", "{'running': True, 'out': 'epoch 3/20 loss 0.91'}"),
        ("print(job_status(job))", "{'running': True, 'out': 'epoch 9/20 loss 0.52'}"),
        ("print(job_status(job))", "{'running': True, 'out': 'epoch 15/20 loss 0.33'}"),
        ("print(job_wait(job))", "{'running': False, 'code': 0, 'out': 'val acc 0.91'}")), True),
    ("Train a small classifier on the dataset and report accuracy.", log(
        ("print(project.sh('python train.py')['out'])", "ModuleNotFoundError: No module named 'sklearn'"),
        ("print(project.sh('python train.py')['out'])", "ModuleNotFoundError: No module named 'sklearn'"),
        ("import importlib; importlib.reload(__import__('os')); print(project.sh('python train.py')['out'])",
         "ModuleNotFoundError: No module named 'sklearn'"),
        ("print(project.sh('python train.py --no-cache')['out'])", "ModuleNotFoundError: No module named 'sklearn'")),
     False),
]

QUESTIONS = {
    "progress": ("Is `work_log` making real progress toward `request` (new results, fixes that work, new files "
                 "or information), rather than repeating the same attempts or going in circles?"),
    "circles": ("Is `work_log` repeating the same or nearly the same attempt and getting the same result, "
                "without learning anything new?"),
    "stuck": "Is the worker stuck: are its recent steps failing or producing nothing new toward `request`?",
}


def run_live() -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for req, wl, _ in CASES:
        r = c.ask({"request": req, "work_log": wl}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    RAW.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    rows = json.loads(RAW.read_text(encoding="utf-8")) if ap.parse_args().offline else run_live()
    labels = [c[2] for c in CASES]
    for (req, _, lab), s in zip(CASES, rows):
        print(f"  {'progress' if lab else 'CIRCLES '}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
              + f"  {req[:50]}")
    for key, stuck_if_high in (("progress", False), ("circles", True), ("stuck", True)):
        print(f"\n== {key}: {QUESTIONS[key][:90]}")
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            flagged = [(s[key] >= t) if stuck_if_high else (s[key] < t) for s in rows]
            caught = sum(f and not l for f, l in zip(flagged, labels))
            false = sum(f and l for f, l in zip(flagged, labels))
            print(f"  t={t:.1f}: stalls caught {caught}/{labels.count(False)}, healthy runs interrupted {false}/{labels.count(True)}")


if __name__ == "__main__":
    main()
