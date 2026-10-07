import re

_SEP = re.compile(r"^:?-+:?$")


def _cells(line):
    line = line.strip()
    parts, cur, i = [], "", 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if c == "|":
            parts.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    parts.append(cur)
    if line.startswith("|"):
        parts = parts[1:]
    if parts and line.endswith("|") and not line.endswith("\\|"):
        parts = parts[:-1]
    return [p.strip() for p in parts]


def _value(s):
    if s == "":
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def parse_table(text):
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if "|" not in lines[i] or "|" not in lines[i + 1]:
            continue
        sep = _cells(lines[i + 1])
        if not sep or not all(_SEP.match(c) for c in sep):
            continue
        head = _cells(lines[i])
        rows = []
        for ln in lines[i + 2:]:
            if not ln.strip() or "|" not in ln:
                break
            cells = _cells(ln)[:len(head)]
            cells += [""] * (len(head) - len(cells))
            rows.append({h: _value(c) for h, c in zip(head, cells)})
        return rows
    return []
