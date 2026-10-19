"""Long-horizon tasks for evals/bench/long_horizon.py: work that cannot be done in a few steps, where what the
worker read early is needed late, and a hidden test says whether it was done right.

`rulebook(seed)` writes a folder of 18 rule files (rules/rule_01.md ...). Each is a page of ordinary-looking
policy prose: one real rule, plus history and rationale paragraphs that contain stale or unrelated numbers (the
2019 pilot used 12%, a retired surcharge...) which must not be mixed up with the rule's own values. Rules are
applied in number order, some amend earlier ones ("Rule 4 no longer applies to pro customers"), and the
worker must write policy.py: decide(case) -> {"total", "status", "flags"}. The expected answers come from a
reference engine that runs the same rule objects the text was written from, so the text and the truth cannot
drift apart. Money is integer cents and percentages are whole numbers, so there is one right answer per case.

The point is the shape of the work: ~18 pages to read (more than one step each, REPL output is capped), then
many steps of writing, testing and fixing, in which any page may be needed again. A worker that keeps what it
read (in a variable, a file, or its context) finishes; one whose context was condensed badly re-reads pages,
mixes up a stale number for a real one, or gives up an exception.

    python evals/bench/long_tasks.py --show 1        # print the rule pages for seed 1
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

REGIONS = ["NORTH", "SOUTH", "EAST", "WEST"]
TIERS = ["basic", "plus", "pro"]
COUPONS = ["SAVE10", "VIP25", "FREESHIP", "BULK5"]
WORDS = ["amber", "kestrel", "lantern", "mosaic", "tundra", "cobalt", "saffron", "harbor", "quartz", "juniper",
         "falcon", "ember", "willow", "granite", "sparrow", "onyx"]

CASE_DOC = """A case is a dict with these keys: region (one of NORTH, SOUTH, EAST, WEST), tier (basic, plus or pro),
amount (the order total in whole cents, an int), items (number of items, an int), express (bool), fragile (bool),
coupon (None or one of SAVE10, VIP25, FREESHIP, BULK5)."""


def pct(total: int, p: int) -> int:
    """p percent of total cents, rounded half up (whole cents)."""
    return (total * p + 50) // 100


@dataclass
class Rule:
    n: int
    kind: str
    p: dict
    exempt: set = field(default_factory=set)       # tiers an amendment takes out of this rule
    amended_by: int | None = None                  # the rule number whose text amends this one


# -- the reference engine -------------------------------------------------------------------------------------------

def applies(r: Rule, case: dict) -> bool:
    return case["tier"] not in r.exempt


def run_rules(rules: list[Rule], case: dict) -> dict:
    total, status, flags = int(case["amount"]), "ok", set()
    for r in rules:
        if r.kind == "amend" or not applies(r, case):
            continue
        k, p = r.kind, r.p
        if status == "reject":
            break
        if k == "handling":              # percent surcharge for a region
            if case["region"] == p["region"]:
                total += pct(total, p["pct"])
        elif k == "tier_discount":       # percent off for a tier with enough items
            if case["tier"] == p["tier"] and case["items"] >= p["items"]:
                total -= pct(total, p["pct"])
        elif k == "flag_over":           # a flag when the current total is over a threshold
            if total > p["cents"]:
                flags.add(p["flag"])
        elif k == "flag_review":         # a flag plus a region sends it to review
            if p["flag"] in flags and case["region"] == p["region"]:
                status = "review"
        elif k == "express":             # flat fee, doubled in one region
            if case["express"]:
                total += p["fee"] * (2 if case["region"] == p["double_in"] else 1)
        elif k == "fragile":             # per-item fee
            if case["fragile"]:
                total += p["per_item"] * case["items"]
        elif k == "coupon":
            if case["coupon"] == p["code"]:
                if p["mode"] == "pct":
                    total -= pct(total, p["pct"])
                elif p["mode"] == "flat":
                    total = max(0, total - p["cents"])
        elif k == "free_ship_flag":      # a coupon sets a flag
            if case["coupon"] == p["code"]:
                flags.add(p["flag"])
        elif k == "cap":                 # a tier's total never exceeds a cap
            if case["tier"] == p["tier"]:
                total = min(total, p["cents"])
        elif k == "floor":               # a minimum total for orders with an express fee
            if case["express"]:
                total = max(total, p["cents"])
        elif k == "reject_over":         # basic customers over a total are rejected (total stays as it is)
            if case["tier"] == p["tier"] and total > p["cents"]:
                status = "reject"
        elif k == "bulk_flag":
            if case["items"] >= p["items"]:
                flags.add(p["flag"])
        elif k == "clear_flag":          # a flag is dropped again in a region
            if case["region"] == p["region"]:
                flags.discard(p["flag"])
    return {"total": total, "status": status, "flags": sorted(flags)}


# -- building the rules from a seed ---------------------------------------------------------------------------------

def build(seed: int) -> list[Rule]:
    rnd = random.Random(seed)
    words = rnd.sample(WORDS, 6)
    regions = rnd.sample(REGIONS, 4)
    rules: list[Rule] = []

    def add(kind, **p):
        rules.append(Rule(len(rules) + 1, kind, p))
        return rules[-1]

    add("handling", region=regions[0], pct=rnd.choice([4, 6, 7, 9]))
    add("tier_discount", tier="plus", items=rnd.choice([3, 4, 5]), pct=rnd.choice([5, 8, 12]))
    add("express", fee=rnd.choice([450, 600, 750, 900]), double_in=regions[1])
    f1 = add("flag_over", cents=rnd.choice([40000, 55000, 72000]), flag=words[0])
    add("flag_review", flag=words[0], region=regions[2])
    add("fragile", per_item=rnd.choice([75, 120, 150, 200]))
    add("coupon", code="SAVE10", mode="pct", pct=rnd.choice([10, 10, 12]))
    add("coupon", code="VIP25", mode="flat", cents=rnd.choice([2500, 3000, 4000]))
    add("free_ship_flag", code="FREESHIP", flag=words[1])
    add("tier_discount", tier="pro", items=rnd.choice([1, 2]), pct=rnd.choice([6, 9, 15]))
    add("handling", region=regions[3], pct=rnd.choice([3, 5, 11]))
    add("bulk_flag", items=rnd.choice([8, 10, 12]), flag=words[2])
    add("cap", tier="basic", cents=rnd.choice([80000, 95000, 120000]))
    add("floor", cents=rnd.choice([1500, 2000, 2500]))
    add("clear_flag", region=regions[0], flag=words[1])
    add("reject_over", tier="basic", cents=rnd.choice([60000, 70000, 85000]))
    # two amendments, written as rules of their own that change an earlier one
    target_a = rules[4]                        # the review rule
    a = add("amend", target=target_a.n, tier="pro")
    target_a.exempt.add("pro")
    target_a.amended_by = a.n
    target_b = rules[5]                        # the fragile fee
    b = add("amend", target=target_b.n, tier="plus")
    target_b.exempt.add("plus")
    target_b.amended_by = b.n
    return rules


# -- the pages --------------------------------------------------------------------------------------------------------

FLUFF = [
    "The rule was first drafted by the operations group after a quarter in which {n} orders needed manual follow-up.",
    "In the 2019 pilot a {p}% figure was used for a similar adjustment; it was retired when the pilot ended and "
    "is not in force.",
    "Finance asked that every change here be reviewed twice a year, and the review notes are kept in the shared drive.",
    "Support reported that customers in the {r} region often asked about this, so the wording was made plainer.",
    "An earlier draft of this memo gave a threshold of {c} cents. That value was superseded and is kept only for the "
    "historical record.",
    "Warehouse staff do not apply this themselves: it is applied by the order system, in rule-number order.",
    "This memo was last proofread in the spring; nothing in the rule itself changed at that time.",
    "Some regional managers prefer a different figure ({p}%) but the head office decided against it for consistency.",
    "The audit team samples about {n} orders a month to check that the rule is applied as written.",
    "Where this memo and an older one disagree, this memo wins.",
    "A temporary surcharge of {p}% applied during the {r} depot move last year and ended with it.",
    "Customer service may quote the old {c}-cent figure to callers who ask; the order system never uses it.",
]
ORDER_NOTE = ("All rules are applied to one order in rule-number order, each one seeing the total, status and flags "
              "left by the rules before it. Percentages are of the current total, rounded half up to a whole cent. "
              "A coupon that no memo mentions does nothing.")


def money(c: int, rnd: random.Random) -> str:
    """The same amount written as cents or as dollars (the order data is always cents)."""
    return f"{c} cents" if rnd.random() < 0.5 else f"${c / 100:,.2f}"


def rule_text(r: Rule, rules: list[Rule], rnd: random.Random | None = None) -> str:
    """The rule as one or two plain sentences, in one of a few wordings (so it cannot be pulled out with one pattern)."""
    rnd = rnd or random.Random(r.n)
    k, p = r.kind, r.p
    v = rnd.randrange(2)
    if k == "handling":
        s = [f"Orders shipped to the {p['region']} region get a handling fee of {p['pct']}% of the current total, added to the total.",
             f"The system adds {p['pct']}% of the running total as handling whenever the destination is {p['region']}."][v]
    elif k == "tier_discount":
        s = [f"A {p['tier']} customer with {p['items']} or more items gets {p['pct']}% off the current total.",
             f"When a {p['tier']}-tier order contains at least {p['items']} items, {p['pct']}% of the running total is taken off."][v]
    elif k == "express":
        s = [f"An express order adds a flat fee of {money(p['fee'], rnd)} to the total, and in the {p['double_in']} region the fee is doubled.",
             f"Express shipping costs a flat {money(p['fee'], rnd)}; for {p['double_in']} destinations it costs twice that."][v]
    elif k == "flag_over":
        s = [f"If the current total is over {money(p['cents'], rnd)} (strictly more), the flag '{p['flag']}' is added to the order.",
             f"An order whose running total goes above {money(p['cents'], rnd)} is tagged '{p['flag']}'; exactly that amount is not above it."][v]
    elif k == "flag_review":
        s = [f"If the order has the flag '{p['flag']}' and ships to the {p['region']} region, its status becomes 'review'. The total is not changed.",
             f"Orders carrying the tag '{p['flag']}' that are headed to {p['region']} have their status set to 'review'; the total stays as it is."][v]
    elif k == "fragile":
        s = [f"A fragile order adds {money(p['per_item'], rnd)} per item to the total.",
             f"For fragile orders, packing costs {money(p['per_item'], rnd)} for each item, added to the total."][v]
    elif k == "coupon" and p["mode"] == "pct":
        s = [f"Coupon {p['code']} takes {p['pct']}% off the current total.",
             f"A customer who enters {p['code']} gets {p['pct']}% of the running total deducted."][v]
    elif k == "coupon":
        s = [f"Coupon {p['code']} takes {money(p['cents'], rnd)} off the current total; the total never goes below zero.",
             f"Entering {p['code']} deducts a flat {money(p['cents'], rnd)}, but never more than the whole total."][v]
    elif k == "free_ship_flag":
        s = [f"Coupon {p['code']} adds the flag '{p['flag']}' to the order. It does not change the total.",
             f"Using {p['code']} marks the order with the tag '{p['flag']}' and leaves the total alone."][v]
    elif k == "bulk_flag":
        s = [f"An order with {p['items']} or more items gets the flag '{p['flag']}'.",
             f"Any order of at least {p['items']} items is tagged '{p['flag']}'."][v]
    elif k == "cap":
        s = [f"A {p['tier']} customer's total is never more than {money(p['cents'], rnd)}: if it is higher, it is lowered to that.",
             f"The most a {p['tier']}-tier customer can be charged is {money(p['cents'], rnd)}; anything above is cut back to it."][v]
    elif k == "floor":
        s = [f"An express order's total is never less than {money(p['cents'], rnd)}: if it is lower, it is raised to that.",
             f"Express orders are charged at least {money(p['cents'], rnd)}; a smaller total is raised to that figure."][v]
    elif k == "clear_flag":
        s = [f"For orders shipped to the {p['region']} region the flag '{p['flag']}' is removed again, if the order has it.",
             f"Orders going to {p['region']} lose the tag '{p['flag']}' if they were given it earlier."][v]
    elif k == "reject_over":
        s = [f"A {p['tier']} customer whose current total is over {money(p['cents'], rnd)} (strictly more) is rejected: the status becomes "
             f"'reject', the total stays as it is, and no later rule changes anything.",
             f"If a {p['tier']}-tier order's running total exceeds {money(p['cents'], rnd)}, the order is rejected and its status is set to 'reject'. Its total is left as "
             f"it is, and nothing after this rule touches it."][v]
    elif k == "amend":
        s = [f"This memo amends Rule {p['target']}: Rule {p['target']} does not apply to {p['tier']} customers, who skip it. "
             f"For everyone else it is unchanged.",
             f"From now on {p['tier']} customers are exempt from Rule {p['target']}; it is skipped for them and works as before for the rest."][v]
    else:  # pragma: no cover
        raise ValueError(k)
    return s


def page(r: Rule, rules: list[Rule], seed: int) -> str:
    """One memo: plain paragraphs, the rule's own sentence somewhere in the middle of them (no headings to find it by)."""
    rnd = random.Random(seed * 1000 + r.n)

    def fl(k: int) -> str:
        return " ".join(f.format(n=rnd.randint(40, 900), p=rnd.choice([2, 3, 5, 12, 14, 18]), r=rnd.choice(REGIONS),
                                 c=rnd.choice([30000, 45000, 65000, 90000])) for f in rnd.sample(FLUFF, k))
    note = f" Rule {r.amended_by} amends this rule, so read that memo as well." if r.amended_by else ""
    return (f"Memo R-{r.n:02d} - order handling\n\n{fl(8)}\n\n{fl(3)} {rule_text(r, rules, rnd)}{note} {ORDER_NOTE}\n\n{fl(9)}\n")


def write_rules(seed: int, folder: Path) -> list[Rule]:
    rules = build(seed)
    (folder / "rules").mkdir(parents=True, exist_ok=True)
    for r in rules:
        (folder / "rules" / f"rule_{r.n:02d}.md").write_text(page(r, rules, seed), encoding="utf-8")
    return rules


def make_cases(seed: int, n: int = 80) -> list[dict]:
    """Random cases, with enough that end in 'review' and 'reject' (and use the amended exceptions) to test those rules."""
    rnd = random.Random(seed + 7)
    rules = build(seed)

    def one() -> dict:
        return {"region": rnd.choice(REGIONS), "tier": rnd.choice(TIERS),
                "amount": rnd.choice([1200, 8800, 19900, 38000, 41000, 56500, 61000, 72500, 91000, 130000]) + rnd.randint(0, 99),
                "items": rnd.choice([1, 1, 2, 3, 4, 5, 8, 10, 12, 15]), "express": rnd.random() < 0.45,
                "fragile": rnd.random() < 0.4, "coupon": rnd.choice([None, None, "SAVE10", "VIP25", "FREESHIP", "BULK5"])}
    out, want = [], {"review": 12, "reject": 12}
    while len(out) < n:
        c = one()
        st = run_rules(rules, c)["status"]
        if st in want:
            if want[st] <= 0:
                continue
            want[st] -= 1
        elif len(out) >= n - sum(max(v, 0) for v in want.values()):
            continue
        out.append(c)
    return out


def hidden_test(seed: int) -> str:
    """The pytest file text: every expected answer comes from the reference engine, one test per case."""
    rules = build(seed)
    cases = make_cases(seed)
    expected = [run_rules(rules, c) for c in cases]
    lines = ["import os, sys", "sys.path.insert(0, os.path.dirname(__file__))", "from policy import decide", "",
             f"CASES = {cases!r}", f"EXPECTED = {expected!r}", "",
             "def _norm(r):", "    return {'total': int(r['total']), 'status': r['status'], 'flags': sorted(r['flags'])}", "",
             "def test_cases():", "    bad = []",
             "    for i, (c, e) in enumerate(zip(CASES, EXPECTED)):",
             "        try:", "            got = _norm(decide(dict(c)))", "        except Exception as x:",
             "            got = repr(x)", "        if got != e:", "            bad.append((i, c, e, got))",
             "    assert not bad, f'{len(bad)} of {len(CASES)} cases wrong, first: {bad[0]}'", ""]
    # also one test per case so the score is the share of cases right
    for i in range(len(cases)):
        lines += [f"def test_case_{i:02d}():", f"    assert _norm(decide(dict(CASES[{i}]))) == EXPECTED[{i}]", ""]
    return "\n".join(lines)


PROMPT = """The folder {dir}/rules holds our order-handling rulebook: rule_01.md ... rule_18.md, one page per rule. Write {dir}/policy.py with a function decide(case) that applies the rules and returns {{"total": int, "status": str, "flags": list}} (flags sorted, status 'ok' unless a rule changes it).

{case_doc}

Read every page: each has background text that is not part of the rule, and some rules amend others. Rules are applied in rule-number order as the pages say. Check decide() against cases you make up yourself, especially the exceptions and the edges (strictly over vs. at the threshold, rounding), then call final().""".replace("{case_doc}", CASE_DOC)


def make_task(seed: int, folder: Path) -> dict:
    write_rules(seed, folder)
    return {"id": f"rulebook{seed}", "prompt": PROMPT.format(dir=folder), "tests": hidden_test(seed), "seed": seed,
            "marks": [f"rule_{n:02d}" for n in range(1, 19)]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, help="print the pages for this seed")
    a = ap.parse_args()
    if a.show is not None:
        for r in build(a.show):
            print(page(r, build(a.show), a.show))
        sys.exit(0)
