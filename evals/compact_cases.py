"""Live eval: when a long task's context is compacted, which earlier steps does the rest of the
task still need in full?

Compaction keeps the last few steps verbatim. Every older step becomes either a one-line note
("ran X -> Y") or, if Jev says the rest of the task still needs it, its code and output in full.
A missed "needed" step costs the worker a step to re-read or re-derive it; a false "needed" only
costs some context. So recall on needed steps matters more than precision.

Cases: (task, where the work is now, the old step's thought/code/output, needed?). The ARC ones
are real steps from evals/bench runs (fd4b2b02, watchtest3); the rest are typical harness work.

    python evals/compact_cases.py            # live, tuning cases
    python evals/compact_cases.py --holdout  # live, held-out cases (check the chosen wording)
    python evals/compact_cases.py --offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevh.config import load_config  # noqa: E402
from jevh.decide import SystemOneClient, noul  # noqa: E402
import _out  # noqa: E402  (evals/_out.py: where a run's results go)

RAW = Path(__file__).with_name("compact_cases_raw.json")
RAW_HOLDOUT = Path(__file__).with_name("compact_holdout_raw.json")

ARC = ("Solve this ARC-AGI puzzle. puzzle.json holds {train: [{input, output}], test: [{input}]}; grids are "
       "lists of rows of colours 0-9. Find the rule, apply it to the test input, save the output grid.")
ARC_NOW = ("- ran: blocks = find_blocks(grid) | for b in blocks: print(b)\n"
           "  -> train 0: block colour 6 rows 5-8 cols 6-7; train 1: block colour 3 rows 2-5 cols 9-12\n"
           "- ran: check_rule(data['train'])\n  -> train 0 ok, train 1 ok, train 2: 4 cells differ")
TESTS = "Fix the failing tests in my project (python, pytest)."
TESTS_NOW = ("- ran: project.edit('app/jobs.py', 'retry(3)', 'retry(attempts)')\n  -> edited app/jobs.py\n"
             "- ran: print(project.sh('pytest -q')['out'])\n  -> 1 failed, 17 passed: test_export_csv - "
             "KeyError: 'amount'")
CSV = "Load sales_2025.csv and report total revenue per month and the top 3 products by revenue."
CSV_NOW = ("- ran: df['month'] = df['order_date'].dt.to_period('M')\n  -> ok\n"
           "- ran: monthly = df.groupby('month')['net_total'].sum(); print(monthly)\n  -> 2025-01 41210.5 ...")
RESEARCH = "Research the best rest times between sets for hypertrophy and summarize the evidence with sources."
RESEARCH_NOW = ("- ran: r3 = web.ask('rest interval hypertrophy trained men 2024 meta-analysis')\n"
                "  -> 2 passages (Singer 2024)\n- ran: draft = compare(notes)\n  -> draft of 3 paragraphs")
SCRAPE = "Scrape product names and prices from the shop's catalogue (5 pages) into products.csv."
SCRAPE_NOW = ("- ran: browser.do('go to the next page'); rows += extract(browser.read())\n  -> done; 36 rows\n"
              "- ran: browser.do('go to the next page'); rows += extract(browser.read())\n  -> done; 48 rows")
COMFY = "Make a hero image for the landing page: a sunrise over mountains, 1920x1080."
COMFY_NOW = ("- ran: wf = comfy.load('qwen_image_t2i'); comfy.set(wf, {'6.text': prompt, '5.width': 1920})\n"
             "  -> ok\n- ran: print(comfy.run(wf))\n  -> RuntimeError: ComfyUI 400: value_not_in_list ckpt_name")
DEPLOY = "Build the site and deploy it to Cloudflare Pages, then give me the live URL."
DEPLOY_NOW = ("- ran: print(project.sh('npx wrangler pages deploy dist --project-name shop')['out'])\n"
              "  -> Deployment complete\n- ran: browser.goto(url); print(browser.read('page title'))\n  -> 'Shop'")
REFACTOR = "Refactor payroll.py into smaller modules and keep the tests green."
REFACTOR_NOW = ("- ran: project.write('payroll/calc.py', calc_src)\n  -> wrote payroll/calc.py\n"
                "- ran: print(project.sh('pytest -q')['out'])\n  -> ImportError: cannot import name 'tax_band'")


def step(thought: str, code: str, output: str) -> str:
    """As jevh.worker renders an old step for this check."""
    return ((f"THOUGHT: ...{thought}\n" if thought else "") + f"CODE:\n{code}\nOUTPUT:\n{output}")


CASES = [  # (task, now, step, needed?)
    # -- ARC (real steps) --
    (ARC, ARC_NOW, step("", "for i, (inp, out) in enumerate(data['train']):\n    print(i, len(inp), len(out))",
                        "0 in 5 x 1 out 6 x 1\nin colors: ['i', 'n', 'p', 't', 'u']"), False),
    (ARC, ARC_NOW, step("that loop iterated over dict keys - print the raw JSON instead",
                        "raw = open('puzzle.json').read()\nprint(raw[:3000])",
                        '{"train": [{"input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '
                        '[0, 0, 0, 0, 0, 0, 0, 0, 0 ...'), False),
    (ARC, ARC_NOW, step("sizes and colours first",
                        "for i, d in enumerate(data['train']):\n    print(i, len(d['input']), len(d['input'][0]))\n"
                        "    print(sorted(set(c for r in d['output'] for c in r)))",
                        "0 in 23 x 23 out 23 x 23\nin colors: [0, 6]\nout colors: [0, 3, 6]\n1 in 16 x 16 out 16 x 16\n"
                        "in colors: [0, 3]\nout colors: [0, 3, 6]\n2 in 26 x 26 out 26 x 26\ntest 16 x 16 colors: [0, 6]"),
     True),
    (ARC, ARC_NOW, step("the output adds colour 3 next to the 6 blocks - where exactly?",
                        "diff = [(r, c, i[r][c], o[r][c]) for r in range(len(i)) for c in range(len(i[0])) "
                        "if i[r][c] != o[r][c]]\nprint(diff[:40])",
                        "[(4, 6, 0, 3), (4, 7, 0, 3), (9, 6, 0, 3), (9, 7, 0, 3), (5, 5, 0, 3), (5, 8, 0, 3), "
                        "(6, 5, 0, 3), (6, 8, 0, 3) ...] 64 cells changed, all 0 -> 3"), True),
    # -- fixing tests --
    (TESTS, TESTS_NOW, step("", "import os\nprint(os.getcwd())", "C:\\Users\\me\\proj"), False),
    (TESTS, TESTS_NOW, step("", "print(project.sh('pytest -q')['out'])",
                            "3 failed, 15 passed\nFAILED test_jobs.py::test_retry - TypeError: retry() takes 1 "
                            "positional argument\nFAILED test_jobs.py::test_backoff\nFAILED test_export.py::"
                            "test_export_csv - KeyError: 'amount'"), False),
    (TESTS, TESTS_NOW, step("the export failure: where does the row dict come from?",
                            "print(project.read('app/export.py', 1, 40))",
                            "    12  def to_rows(records):\n    13      return [{'id': r.id, 'total': r.total, 'date': "
                            "r.date} for r in records]\n    ...\n    31  def write_csv(rows, path):\n    32      "
                            "w.writerow([row['id'], row['amount'], row['date']])"), True),
    (TESTS, TESTS_NOW, step("", "print(project.sh('pip list')['out'][:2000])",
                            "Package    Version\n---------- -------\nattrs 23.2.0\npytest 8.3.2\nrequests 2.32.3 ..."),
     False),
    # -- CSV analysis --
    (CSV, CSV_NOW, step("", "df = pd.read_csv('sales_2025.csv')",
                        "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 1882"), False),
    (CSV, CSV_NOW, step("latin-1 then; look at the columns",
                        "df = pd.read_csv('sales_2025.csv', encoding='latin-1')\nprint(df.dtypes)",
                        "order_id int64\norder_date datetime64[ns]\nsku object\nproduct_name object\nqty int64\n"
                        "unit_price float64\nnet_total float64\nrefunded bool"), True),
    (CSV, CSV_NOW, step("refunded orders should not count as revenue",
                        "print(df['refunded'].sum(), len(df)); df = df[~df['refunded']]",
                        "312 18450"), True),
    (CSV, CSV_NOW, step("", "print(pd.__version__)", "2.2.3"), False),
    # -- research --
    (RESEARCH, RESEARCH_NOW, step("", "r = web.ask('rest between sets hypertrophy evidence')\nprint(r)",
                                  "{'answer': 'Longer rests (2-3 min) led to greater hypertrophy than 1 min in "
                                  "trained men', 'sources': ['https://pubmed.ncbi.nlm.nih.gov/26605807 (Schoenfeld "
                                  "2016)']}"), True),
    (RESEARCH, RESEARCH_NOW, step("", "r2 = web.ask('rest interval hypertrophy women')\nprint(r2)",
                                  "{'answer': 'No relevant passages found.', 'sources': []}"), False),
    # -- scraping --
    (SCRAPE, SCRAPE_NOW, step("", "print(browser.goto('https://shop.example.com'))", "{'status': 'done'}"), False),
    (SCRAPE, SCRAPE_NOW, step("the product cards: what markup do they use?",
                              "print(browser.read('the HTML structure of one product card')[:1500])",
                              "<div class='card'><h3 class='p-name'>Trail Runner 2</h3><span class='price'>"
                              "$89.00</span>...  extract() uses .p-name and .price"), True),
    # -- ComfyUI --
    (COMFY, COMFY_NOW, step("which checkpoints are installed?", "print(comfy.models('checkpoints'))",
                            "['Qwen/qwen_image_2.1_fp8.safetensors', 'sdxl/juggernautXL_v9.safetensors', "
                            "'flux1-dev-fp8.safetensors']"), True),
    (COMFY, COMFY_NOW, step("", "print(comfy.ping())", "True"), False),
    # -- deploy --
    (DEPLOY, DEPLOY_NOW, step("", "print(project.sh('npm run build')['out'])",
                              "Error: VITE_API_URL is not defined"), False),
    (DEPLOY, DEPLOY_NOW, step("", "r = project.sh('npx wrangler pages deploy dist --project-name shop')\nprint(r)",
                              "Deployment complete! Take a peek over at https://4f2a91c3.shop-7rq.pages.dev\n"
                              "Alias: https://shop-7rq.pages.dev"), True),
    # -- refactor --
    (REFACTOR, REFACTOR_NOW, step("list what's in payroll.py before splitting it",
                                  "print(project.grep(r'^def |^class ', 'payroll.py'))",
                                  "payroll.py:12: def gross(hours, rate)\npayroll.py:40: def tax_band(income)\n"
                                  "payroll.py:77: def net(gross, band)\npayroll.py:120: def export(rows, path)\n"
                                  "payroll.py:160: class Employee"), True),
    (REFACTOR, REFACTOR_NOW, step("", "print(project.tree())", "payroll.py\ntests/\n  test_payroll.py\nREADME.md"),
     False),
    (REFACTOR, REFACTOR_NOW, step("", "print(project.sh('git status --short')['out'])", "(clean)"), False),
    # -- a lesson from a failed attempt --
    (ARC, ARC_NOW, step("so mirroring is wrong: train 2 has blocks touching the edge and the 3s wrap",
                        "out = mirror_rule(data['train'][2]['input'])\nprint(out == data['train'][2]['output'])",
                        "False (18 cells differ: all at the grid edge, where the 3s go on the opposite side)"), True),
    (TESTS, TESTS_NOW, step("", "print(project.sh('pytest -q tests/test_jobs.py')['out'])",
                            "2 failed: test_retry, test_backoff - TypeError: retry() takes 1 positional argument"),
     False),
    (CSV, CSV_NOW, step("", "print(df.head(3))",
                        "   order_id order_date     sku  product_name  qty ...\n0  10001 2025-01-02  TR-2  Trail "
                        "Runner 2  1 ..."), False),
]

HOLDOUT = [  # checked once, after the wording and threshold were picked on CASES
    (ARC, ARC_NOW, step("print each grid as digits so I can see the shapes",
                        "def show(g):\n    for r in g: print(''.join(map(str, r)))\nfor d in data['train']: "
                        "show(d['input']); show(d['output'])",
                        "00000000000000000000000\n00000066000000000000000\n00000066000000000000000\n..."), True),
    (ARC, ARC_NOW, step("", "import numpy as np\nprint(np.__version__)", "2.1.1"), False),
    (TESTS, TESTS_NOW, step("what does the test expect?", "print(project.read('tests/test_export.py', 1, 30))",
                            "    8  def test_export_csv(tmp_path):\n    9      rows = [{'id': 1, 'total': 9.5, 'date': "
                            "'2025-01-01'}]\n   10      write_csv(rows, tmp_path / 'o.csv')\n   11      assert "
                            "'9.5' in (tmp_path / 'o.csv').read_text()"), True),
    (TESTS, TESTS_NOW, step("", "print(project.read('README.md', 1, 20))",
                            "    1  # proj\n    2  Small job runner. Run `pytest` to test."), False),
    (CSV, CSV_NOW, step("which products earn the most?",
                        "top = df.groupby('product_name')['net_total'].sum().nlargest(3); print(top)",
                        "Trail Runner 2   58210.0\nPeak Jacket   41990.5\nBase Layer Set   30117.0"), True),
    (CSV, CSV_NOW, step("", "df = pd.read_csv('sales_2025.csv', encoding='utf-16')", "UnicodeError: UTF-16 stream "
                        "does not start with BOM"), False),
    (RESEARCH, RESEARCH_NOW, step("", "r4 = web.ask('rest interval strength vs hypertrophy review')\nprint(r4)",
                                  "{'answer': 'A 2017 review (Grgic) found rests over 60 s better for hypertrophy "
                                  "with multi-joint lifts', 'sources': ['https://pubmed.ncbi.nlm.nih.gov/28933024']}"),
     True),
    (SCRAPE, SCRAPE_NOW, step("", "print(browser.do('accept the cookie banner'))", "{'status': 'done'}"), False),
    (SCRAPE, SCRAPE_NOW, step("how many pages are there?", "print(browser.read('the pagination links'))",
                              "Pages: 1 2 3 4 5 Next. Page 5 is the last; URLs /catalogue?page=N"), True),
    (COMFY, COMFY_NOW, step("", "print(comfy.run(wf))", "RuntimeError: ComfyUI 400: value_not_in_list ckpt_name "
                            "'qwen.safetensors'"), False),
    (DEPLOY, DEPLOY_NOW, step("", "print(project.sh('node -v')['out'])", "v22.4.0"), False),
    (DEPLOY, DEPLOY_NOW, step("the build needs the API URL", "project.write('.env.production', "
                              "'VITE_API_URL=https://api.shop.example.com')", "wrote .env.production"), True),
    (REFACTOR, REFACTOR_NOW, step("who imports tax_band?", "print(project.grep('tax_band'))",
                                  "payroll.py:40: def tax_band(income)\nreports/yearly.py:3: from payroll import "
                                  "tax_band, net\ntests/test_payroll.py:5: from payroll import tax_band"), True),
    (REFACTOR, REFACTOR_NOW, step("", "print(project.sh('python --version')['out'])", "Python 3.12.4"), False),
]

QUESTIONS = {
    "needed": ("Will finishing `task` from where `now` is still need what `step` found or did - a value, name, "
               "structure, result, file, error or conclusion that later work relies on - and not something "
               "superseded, fixed since, or of no use to the task?"),
    "info": ("Does `step` hold information that finishing `task` will use again, such as data values, "
             "column or function names, file contents, sources, URLs or a finding about what does or does not work?"),
    "useless": ("Is `step` of no further use for `task`: a failed attempt that was fixed since, output superseded by "
                "later steps, or a check that told the task nothing?"),
}


def run_live(cases, raw: Path) -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for task, now, st, _ in cases:
        r = c.ask({"task": task, "now": now, "step": st}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(cases, rows) -> None:
    labels = [c[3] for c in cases]
    for (task, _, st, lab), s in zip(cases, rows):
        print(f"  {'NEEDED' if lab else 'drop  '}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
              + f"  {task[:18]} | {st.split('CODE:')[-1].strip()[:50]!r}")
    for key in QUESTIONS:
        keep_if_high = key != "useless"
        print(f"\n== {key}: {QUESTIONS[key][:90]}")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            kept = [(s[key] >= t) if keep_if_high else (s[key] < t) for s in rows]
            hit = sum(k and l for k, l in zip(kept, labels))
            extra = sum(k and not l for k, l in zip(kept, labels))
            print(f"  t={t:.1f}: needed kept {hit}/{labels.count(True)}, needless kept {extra}/{labels.count(False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    _out.add_args(ap)
    a = ap.parse_args()
    dest = _out.Out.of(a)
    cases, raw = (HOLDOUT, RAW_HOLDOUT) if a.holdout else (CASES, RAW)
    if a.offline:
        rows = json.loads(dest.read(raw).read_text(encoding="utf-8"))
    else:
        rows = run_live(cases, dest.write(raw))
        print(f"raw scores saved to {dest.where(dest.write(raw))}")
    report(cases, rows)


if __name__ == "__main__":
    main()
