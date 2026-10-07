def _m(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _s(n):
    return f"{n // 60:02d}:{n % 60:02d}"


def first_free(busy, duration, day_start, day_end):
    lo, hi = _m(day_start), _m(day_end)
    iv = sorted((_m(a), _m(b)) for person in busy for a, b in person)
    t = lo
    for a, b in iv:
        if a >= t + duration:
            break
        if b > t:
            t = b
    return _s(t) if t + duration <= hi else None
