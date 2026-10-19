"""Live eval: the cheap first pass - is a code change risky enough for a full review?

When a worker changed project files, Jev asks a few yes/no risk questions per changed file.
Risky -> the strong model reviews the diff; safe -> accepted on the tests alone. A missed risky
change skips its review (the costly error); a safe change flagged only costs a review's time.
So: recall on risky first, then as few safe ones flagged as possible.

    python evals/risk_cases.py            # live, tuning cases
    python evals/risk_cases.py --holdout  # live, held-out cases
    python evals/risk_cases.py --offline
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

RAW = Path(__file__).with_name("risk_cases_raw.json")
RAW_HOLDOUT = Path(__file__).with_name("risk_holdout_raw.json")


def diff(old: str, new: str) -> str:
    return "\n".join(["-" + ln for ln in old.splitlines()] + ["+" + ln for ln in new.splitlines()])


CASES = [  # (request, file, change, risky?)
    # -- risky --
    ("Make the export endpoint faster.", "app/routes/export.py",
     diff("@app.get('/export')\n@require_auth\ndef export():", "@app.get('/export')\ndef export():"), True),
    ("Let team leads edit their team's projects.", "app/permissions.py",
     diff("    return user.is_admin", "    return user.is_admin or user.role == 'lead' or project.team_id"), True),
    ("Add a nightly cleanup of old orders.", "app/jobs/cleanup.py",
     diff("", "def cleanup(db):\n    db.execute('DELETE FROM orders WHERE created_at < ?', (cutoff(),))"), True),
    ("We don't use the fax column anymore.", "migrations/0014_drop_fax.py",
     diff("", "def upgrade():\n    op.drop_column('customers', 'fax')"), True),
    ("Clear the cache on startup.", "app/startup.py",
     diff("", "shutil.rmtree(settings.DATA_DIR / 'cache')"), True),
    ("Speed up login.", "app/auth.py",
     diff("    return bcrypt.checkpw(pw.encode(), user.pw_hash)",
          "    return hashlib.md5(pw.encode()).hexdigest() == user.pw_hash"), True),
    ("Add search by name to the customers page.", "app/views/customers.py",
     diff("", "    q = request.args['q']\n    rows = db.execute(f\"SELECT * FROM customers WHERE name LIKE '%{q}%'\")"),
     True),
    ("Let users convert uploaded images to PNG.", "app/convert.py",
     diff("", "def to_png(filename):\n    subprocess.run(f'magick {filename} out.png', shell=True)"), True),
    ("Charges are off by a cent sometimes; fix it.", "app/billing.py",
     diff("    cents = int(amount * 100)", "    cents = round(amount * 100)"), True),
    ("Rename the id parameter to user_id for clarity.", "lib/users.py",
     diff("def get_user(id):", "def get_user(user_id):"), True),
    ("Use camelCase in the orders API response.", "app/routes/orders.py",
     diff("    return jsonify(order_id=o.id, total=o.total)", "    return jsonify(orderId=o.id, total=o.total)"),
     True),
    ("Show comments with their formatting.", "app/templates/post.html",
     diff("  <p>{{ comment.body }}</p>", "  <p>{{ comment.body | safe }}</p>"), True),
    ("The counter is slow; simplify it.", "app/counter.py",
     diff("    with self._lock:\n        self.n += 1", "    self.n += 1"), True),
    # -- safe --
    ("Fix the typo in the docstring.", "app/billing.py",
     diff('    """Retrun the invoice total."""', '    """Return the invoice total."""'), False),
    ("Log when a sync finishes.", "src/sync.py",
     diff("    return rows", "    log.info('synced %d rows', rows)\n    return rows"), False),
    ("Tidy up sync.py.", "src/sync.py", diff("    tmp = fetch(uid)\n    rows += len(tmp)",
                                             "    orders = fetch(uid)\n    rows += len(orders)"), False),
    ("Add a test for the total.", "tests/test_billing.py",
     diff("", "def test_total():\n    assert total([1.5, 2.5]) == 4.0"), False),
    ("Make the buttons blue.", "static/site.css", diff("button { background: #333; }",
                                                       "button { background: #2563eb; }"), False),
    ("Document how to run the app.", "README.md", diff("", "## Running\n\n    python -m app\n"), False),
    ("Split the long function.", "app/report.py",
     diff("", "def _row(o):\n    return [o.id, o.total, o.status]\n"), False),
    ("Add type hints to money.py.", "lib/money.py",
     diff("def to_cents(amount):", "def to_cents(amount: float) -> int:"), False),
    ("Make the error message clearer.", "app/views/upload.py",
     diff("    flash('Error')", "    flash('That file is too large (max 10 MB).')"), False),
    ("Change the page title.", "app/templates/base.html",
     diff("  <title>App</title>", "  <title>Orders - Shop</title>"), False),
    ("Reject empty names in the signup form.", "app/forms.py",
     diff("", "    if not name.strip():\n        raise ValidationError('Name is required')"), False),
    ("Add a helper to format dates in the report.", "app/report.py",
     diff("", "def _fmt(d):\n    return d.strftime('%Y-%m-%d')"), False),
]

HOLDOUT = [
    ("Allow password reset without the email step for now.", "app/auth/reset.py",
     diff("    if not verify_email_token(token):\n        abort(403)", ""), True),
    ("Remove old user uploads.", "app/jobs/purge.py",
     diff("", "for f in Path(UPLOADS).glob('*'):\n    f.unlink()"), True),
    ("Read the API key for the mailer.", "app/mailer.py",
     diff("MAIL_KEY = os.environ['MAIL_KEY']", "MAIL_KEY = 'mk_live_83hf8s7df6'"), True),
    ("Support redirects after login.", "app/views/login.py",
     diff("    return redirect(url_for('home'))", "    return redirect(request.args.get('next', '/'))"), True),
    ("Change the date format in the CSV export.", "app/export.py",
     diff("    row['date'] = d.isoformat()", "    row['date'] = d.strftime('%m/%d/%Y')"), True),
    ("Make refunds partial.", "app/billing.py",
     diff("def refund(order):\n    stripe.Refund.create(charge=order.charge_id)",
          "def refund(order, amount):\n    stripe.Refund.create(charge=order.charge_id, amount=amount)"), True),
    ("Fix the spelling in the footer.", "app/templates/footer.html",
     diff("  <p>Copywrite 2026</p>", "  <p>Copyright 2026</p>"), False),
    ("Add a test for refunds.", "tests/test_refund.py",
     diff("", "def test_refund_calls_stripe(mock_stripe):\n    refund(order)\n    mock_stripe.assert_called()"),
     False),
    ("Add spacing to the cards.", "static/cards.css", diff(".card { margin: 0; }", ".card { margin: 12px; }"),
     False),
    ("Log slow requests.", "app/middleware.py",
     diff("", "    if elapsed > 1.0:\n        log.warning('slow request %s %.2fs', path, elapsed)"), False),
    ("Rename a local variable.", "app/views/orders.py",
     diff("    x = repo.orders_for(uid)\n    return render('o.html', orders=x)",
          "    orders = repo.orders_for(uid)\n    return render('o.html', orders=orders)"), False),
    ("Add a docstring to the repo module.", "app/repo.py",
     diff("", '"""Database access for orders and customers."""'), False),
]

QUESTIONS = {
    "access": "Does `change` affect login, permissions, or who is allowed to see or do what?",
    "data": "Could `change` delete, overwrite or corrupt stored data or files, or change a database schema?",
    "unsafe": ("Does `change` handle secrets, passwords, money or payments, or pass untrusted input into SQL, "
               "shell commands, file paths, redirects or HTML?"),
    "interface": ("Does `change` alter something other code or users rely on - a function's parameters, an API "
                  "response, a file or data format - or remove a safeguard such as a lock or a check?"),
    "risky": ("Could `change` cause a security problem, data loss or a breaking change if it has a mistake - "
              "rather than being a small safe change like docs, styling, logging, tests or a local rename?"),
}
SPECIFIC = ["access", "data", "unsafe", "interface"]


def run_live(cases, raw: Path) -> list[dict]:
    cfg = load_config()
    c = SystemOneClient(cfg.jev["url"], cfg.jev["backend"], timeout=300)
    rows = []
    for req, file, change, _ in cases:
        r = c.ask({"request": req, "file": file, "change": change}, {k: noul(q) for k, q in QUESTIONS.items()})
        rows.append({k: r[k].noul for k in QUESTIONS})
    raw.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(cases, rows) -> None:
    labels = [c[3] for c in cases]
    for (req, file, _, lab), s in zip(cases, rows):
        print(f"  {'RISKY' if lab else 'safe '}  " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
              + f"  {file[:26]:26} | {req[:40]}")

    def show(name, flagged):
        hit = sum(f and l for f, l in zip(flagged, labels))
        extra = sum(f and not l for f, l in zip(flagged, labels))
        return f"{name}: risky caught {hit}/{labels.count(True)}, safe flagged {extra}/{labels.count(False)}"
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print("  " + show(f"any specific >= {t:.1f}", [max(s[k] for k in SPECIFIC) >= t for s in rows]))
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print("  " + show(f"risky >= {t:.1f}", [s["risky"] >= t for s in rows]))


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
