"""Small text statistics helpers."""
import re

_ABBR = ("dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "e.g", "i.e", "vs", "etc")


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower())


def top_n(counts, n):
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def sentences(text):
    text = text.strip()
    out, start = [], 0
    for m in re.finditer(r"[.!?]\s+(?=[A-Z0-9])", text):
        chunk = text[start:m.start() + 1]
        last = chunk.split()[-1].lower().rstrip(".!?") if chunk.split() else ""
        if m.group()[0] == "." and last in _ABBR:
            continue
        out.append(chunk)
        start = m.end()
    out.append(text[start:])
    return [p for p in out if p]


def wrap(text, width):
    lines, cur = [], ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
