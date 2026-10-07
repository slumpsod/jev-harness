import csv
import re
from datetime import datetime


def _amount(s):
    s = s.strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").strip()
    v = float(s)
    return -abs(v) if neg else v


def summarize(path):
    seen, months = set(), {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ("date", "description", "amount", "currency"))
            if key in seen:
                continue
            seen.add(key)
            amt = _amount(row["amount"])
            if amt is None:
                continue
            d = row["date"].strip()
            dt = datetime.strptime(d, "%Y-%m-%d") if re.match(r"\d{4}-", d) else datetime.strptime(d, "%d/%m/%Y")
            m = months.setdefault(f"{dt.year}-{dt.month:02d}", {"in": 0.0, "out": 0.0})
            m["in" if amt > 0 else "out"] += abs(amt)
    return {k: {"in": round(v["in"], 2), "out": round(v["out"], 2), "net": round(v["in"] - v["out"], 2)}
            for k, v in sorted(months.items())}
