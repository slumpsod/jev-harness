"""The orb's chat, drawn: a slim glassy bar to type in, and speech bubbles that come out of the orb (jevh/orb.py shows
them in layered windows, so they are see-through around their shapes, with soft shadows, like the orb).

Pure Pillow (fonts from Windows when there, Pillow's own otherwise), so it is tested and previewed without a screen.
Each draw returns a straight-alpha RGBA image and the regions that can be clicked in it (name -> box).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
PANEL = (19, 25, 35)            # the glass: deep blue-grey ...
PANEL_A = 236                   # ... nearly opaque at the default glass, so text reads on any wallpaper
GLASS = 10                      # how glassy the chat is, 0 (solid) to 100: the Settings tab sets it (appearance.orb_glass)
EDGE = (110, 205, 255)          # a thin water-blue rim, the orb's colour
TEXT, MUTED, LINK = (232, 238, 246), (140, 152, 170), (120, 205, 255)
ACCENT, AMBER = (64, 170, 245), (255, 176, 72)
CODE_BG = (33, 41, 56)
BAR_H, BUBBLE_W, BUBBLE_MAX_H, PAD = 40, 340, 330, 14


@lru_cache(maxsize=32)
def font(size: int, kind: str = "text"):
    names = {"text": ["segoeui.ttf"], "bold": ["segoeuib.ttf", "segoeui.ttf"], "mono": ["consola.ttf", "cour.ttf"],
             "emoji": ["seguiemj.ttf"]}[kind]
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, n), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:   # (an old Pillow)
        return ImageFont.load_default()


def _shadow(size, box, radius, blur, alpha=110):
    layer = Image.new("L", size, 0)
    ImageDraw.Draw(layer).rounded_rectangle(box, radius, fill=alpha)
    return layer.filter(ImageFilter.GaussianBlur(blur))


def tint_alpha(glass: float) -> int:
    """How opaque the glass's tint is: 0 -> 250 (solid), the default 10 -> 236, 100 -> 110 (more of what is behind shows;
    jevh/orb.py frosts that, so text still reads)."""
    g = min(100.0, max(0.0, float(glass))) / 100
    return round(250 - 140 * g)


def _compose(size, shape, shadow, fill_rgb, fill_a, edge_a=70, glass=GLASS):
    """A shape (an L mask) with its shadow underneath and a thin rim of the orb's blue. The more glass, the stronger the
    sheen across its top and the brighter its rim. The shape is kept in img.info["shape"] (jevh/orb.py frosts under it)."""
    g = min(100.0, max(0.0, float(glass))) / 100
    edge_a = round(edge_a + 90 * g)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    sh = Image.new("RGBA", size, (0, 0, 0, 0))
    sh.putalpha(shadow)
    img.alpha_composite(sh)
    body = Image.new("RGBA", size, fill_rgb + (0,))
    body.putalpha(shape.point(lambda v: v * fill_a // 255))
    img.alpha_composite(body)
    rim = shape.filter(ImageFilter.MaxFilter(3))
    edge = Image.eval(Image.fromarray(np.clip(np.asarray(rim, np.int16) - np.asarray(shape.filter(ImageFilter.MinFilter(3)), np.int16), 0, 255).astype(np.uint8)), lambda v: v * edge_a // 255)
    e = Image.new("RGBA", size, EDGE + (0,))
    e.putalpha(edge)
    img.alpha_composite(e)
    if g > 0.15:   # the sheen: light on the top of the glass, fading out by its middle
        ys, _ = np.nonzero(np.asarray(shape) > 127)
        if len(ys):
            top, bot = ys.min(), ys.max()
            ramp = np.clip(1 - (np.arange(size[1]) - top) / max(1, (bot - top) * 0.55), 0, 1)
            a = (np.asarray(shape, np.float32) / 255) * ramp[:, None] * (34 * g)
            sheen = Image.new("RGBA", size, (255, 255, 255, 0))
            sheen.putalpha(Image.fromarray(a.astype(np.uint8)))
            img.alpha_composite(sheen)
    img.info["shape"] = shape
    return img


# ---------------------------------------------------------------------- the bar
def draw_bar(text: str, width: int, scale: float = 1.0, caret: bool = True, caret_at: int | None = None,
             secret: bool = False, placeholder: str = "Ask anything...", busy: bool = False, glass: float = GLASS):
    """The input bar, `width` px wide (it slides out: any width from its height up). -> (image, regions)"""
    s = scale
    h, m = round(BAR_H * s), round(10 * s)                 # m: room for the shadow
    W, H = max(width, h) + 2 * m, h + 2 * m
    box = (m, m, m + max(width, h) - 1, m + h - 1)
    shape = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shape).rounded_rectangle(box, h // 2, fill=255)
    img = _compose((W, H), shape, _shadow((W, H), (m + 2, m + 4, box[2] - 2, box[3] + 3), h // 2, 6 * s), PANEL, tint_alpha(glass),
                   glass=glass)
    d = ImageDraw.Draw(img)
    regions = {}
    f = font(round(14 * s))
    send_r = round(14 * s)
    sx, sy = box[2] - round(6 * s) - send_r, (box[1] + box[3]) // 2
    room = box[2] - box[0] - h // 2 - 2 * send_r - round(18 * s)
    if room > 20:   # (still sliding out: no text yet)
        shown = ("\u2022" * len(text)) if secret else text
        x0, y = box[0] + h // 2, (box[1] + box[3]) // 2
        if shown:
            caret_at = len(shown) if caret_at is None else caret_at
            # keep the caret in view: drop characters from the left until the part before it fits
            start = 0
            while start < caret_at and f.getlength(shown[start:caret_at]) > room - 4:
                start += 1
            vis = shown[start:]
            while vis and f.getlength(vis) > room:
                vis = vis[:-1]
            d.text((x0, y), vis, font=f, fill=TEXT, anchor="lm")
            cx = x0 + f.getlength(shown[start:caret_at])
        else:
            d.text((x0, y), placeholder, font=f, fill=MUTED, anchor="lm")
            cx = x0
        if caret:
            d.line((cx + 1, y - 9 * s, cx + 1, y + 9 * s), fill=LINK, width=max(1, round(1.4 * s)))
        # the send button: a filled circle with an arrow once there is something to send; dots while it works
        if busy:
            for i in range(3):
                d.ellipse((sx - 9 * s + i * 9 * s - 2 * s, sy - 2 * s, sx - 9 * s + i * 9 * s + 2 * s, sy + 2 * s), fill=LINK)
        else:
            on = bool(text)
            d.ellipse((sx - send_r, sy - send_r, sx + send_r, sy + send_r), fill=ACCENT if on else (44, 54, 70))
            a = 6 * s
            col = (255, 255, 255) if on else MUTED
            d.line((sx, sy + a, sx, sy - a), fill=col, width=max(1, round(2 * s)))
            d.line((sx - a * 0.8, sy - a * 0.2, sx, sy - a, sx + a * 0.8, sy - a * 0.2), fill=col, width=max(1, round(2 * s)),
                   joint="curve")
        regions["send"] = (sx - send_r, sy - send_r, sx + send_r, sy + send_r)
    regions["bar"] = box
    return img, regions


# ---------------------------------------------------------------------- the speech bubble
def _plain(text: str) -> list[tuple[str, str]]:
    """Markdown, roughly: -> [(kind, line)], kind = text | bullet | code | gap. Bold and backticks are dropped (the
    words stay); code blocks are kept as code lines."""
    out, code = [], False
    for raw in str(text or "").replace("\r", "").split("\n"):
        if raw.strip().startswith("```"):
            code = not code
            continue
        if code:
            out.append(("code", raw.rstrip()))
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"^#+\s*", "", line)
        if not line.strip():
            out.append(("gap", ""))
        elif re.match(r"^\s*[-*]\s+", line):
            out.append(("bullet", re.sub(r"^\s*[-*]\s+", "", line)))
        else:
            out.append(("text", line.strip()))
    while out and out[-1][0] == "gap":
        out.pop()
    return out


def _wrap(text: str, f, width: float) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        nxt = f"{cur} {w}" if cur else w
        if f.getlength(nxt) <= width:
            cur = nxt
            continue
        if cur:
            lines.append(cur)
        while f.getlength(w) > width:   # a word longer than the line (a URL): cut it
            k = len(w)
            while k > 1 and f.getlength(w[:k]) > width:
                k -= 1
            lines.append(w[:k])
            w = w[k:]
        cur = w
    lines.append(cur)
    return lines


def layout(model: dict, scale: float = 1.0) -> list[tuple]:
    """The bubble's content as drawable rows: (kind, text, font, height)."""
    s = scale
    f, fb, fm, fs = font(round(14 * s)), font(round(14 * s), "bold"), font(round(12.5 * s), "mono"), font(round(12 * s))
    inner = round((BUBBLE_W - 2 * PAD) * s)
    rows = []
    kind = model.get("kind")
    if kind == "waiting":
        rows.append(("status", model.get("status") or "Working...", f, round(22 * s)))
        return rows
    if kind == "question":
        rows.append(("label", "Asking you", fs, round(18 * s)))
    if model.get("you"):
        for line in _wrap(model["you"], fs, inner)[:2]:
            rows.append(("you", line, fs, round(18 * s)))
        rows.append(("gap", "", f, round(6 * s)))
    for k, line in _plain(model.get("text", "")):
        if k == "gap":
            rows.append(("gap", "", f, round(8 * s)))
        elif k == "code":
            for part in _wrap(line or " ", fm, inner - 12 * s) or [""]:
                rows.append(("code", part, fm, round(18 * s)))
        elif k == "bullet":
            for i, part in enumerate(_wrap(line, f, inner - 14 * s)):
                rows.append(("bullet" if i == 0 else "indent", part, f, round(20 * s)))
        else:
            for part in _wrap(line, fb if kind == "question" else f, inner):
                rows.append(("text", part, fb if kind == "question" else f, round(20 * s)))
    return rows


def draw_bubble(model: dict, scale: float = 1.0, tail: str = "right", below: bool = False, scroll: int = 0,
                tick: int = 0, glass: float = GLASS):
    """A speech bubble, its tail pointing at the orb (tail: the side the orb is on; below: the bubble is under the orb).
    model: {kind: reply | waiting | question | error, text, you (what was asked), options: [...], persona}.
    -> (image, regions, overflow): regions: close, open, option:<i>; overflow: how far it scrolls."""
    s = scale
    rows = layout(model, s)
    options = list(model.get("options") or [])[:6] if model.get("kind") == "question" else []
    fo, fs = font(round(13 * s)), font(round(12 * s))
    content_h = sum(r[3] for r in rows)
    opt_h = 0
    opt_rows: list[list[tuple[str, int]]] = []
    if options:   # option pills, flowing onto as many lines as they need
        inner = round((BUBBLE_W - 2 * PAD) * s)
        line, used = [], 0
        for i, o in enumerate(options):
            w = round(fo.getlength(o) + 24 * s)
            if line and used + w > inner:
                opt_rows.append(line)
                line, used = [], 0
            line.append((o, i))
            used += w + round(6 * s)
        opt_rows.append(line)
        opt_h = len(opt_rows) * round(34 * s) + round(4 * s)
    foot_h = 0 if model.get("kind") == "waiting" else round(26 * s)
    view_h = min(content_h, round(BUBBLE_MAX_H * s) - opt_h - foot_h)
    overflow = max(0, content_h - view_h)
    scroll = max(0, min(scroll, overflow))
    bw = round(BUBBLE_W * s)
    if content_h < view_h + 1 and not options:   # a short reply: as wide as it needs
        widest = max([r[2].getlength(r[1]) + (14 * s if r[0] in ("bullet", "indent") else 0) for r in rows] + [120 * s])
        bw = min(bw, round(widest + 2 * PAD * s + 28 * s))
    bh = round(PAD * s) + view_h + opt_h + foot_h + round(4 * s)
    m, tl = round(14 * s), round(12 * s)                      # margin for the shadow, tail length
    W, H = bw + 2 * m, bh + 2 * m + tl
    top = m + (tl if below else 0)
    box = (m, top, m + bw - 1, top + bh - 1)
    shape = Image.new("L", (W, H), 0)
    ds = ImageDraw.Draw(shape)
    r = round(16 * s)
    ds.rounded_rectangle(box, r, fill=255)
    tx = box[2] - round(30 * s) if tail == "right" else box[0] + round(30 * s)
    if below:
        ds.polygon([(tx - 9 * s, box[1] + 1), (tx + 9 * s, box[1] + 1), (tx + (8 if tail == "right" else -8) * s, box[1] - tl)], fill=255)
    else:
        ds.polygon([(tx - 9 * s, box[3] - 1), (tx + 9 * s, box[3] - 1), (tx + (8 if tail == "right" else -8) * s, box[3] + tl)], fill=255)
    shadow = shape.filter(ImageFilter.GaussianBlur(7 * s)).point(lambda v: v * 120 // 255)
    shadow = Image.fromarray(np.roll(np.asarray(shadow), round(4 * s), axis=0))
    img = _compose((W, H), shape, shadow, PANEL, tint_alpha(glass), glass=glass)
    d = ImageDraw.Draw(img)
    regions = {}
    # the content, scrolled, clipped to the view
    layer = Image.new("RGBA", (bw, max(1, content_h)), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    y = 0
    x0 = round(PAD * s)
    for kind, text, f, rh in rows:
        if kind == "status":
            dots = "." * (1 + tick % 3)
            dl.text((x0, y + rh // 2), text.rstrip(".") + dots, font=f, fill=MUTED, anchor="lm")
        elif kind == "label":
            dl.text((x0, y + rh // 2), text.upper(), font=f, fill=AMBER, anchor="lm")
        elif kind == "you":
            dl.text((x0, y + rh // 2), text, font=f, fill=MUTED, anchor="lm")
        elif kind == "code":
            dl.rectangle((x0 - 4 * s, y, bw - x0 + 4 * s, y + rh), fill=CODE_BG + (255,))
            dl.text((x0 + 2 * s, y + rh // 2), text, font=f, fill=TEXT, anchor="lm")
        elif kind == "bullet":
            dl.ellipse((x0 + 2 * s, y + rh // 2 - 2 * s, x0 + 6 * s, y + rh // 2 + 2 * s), fill=LINK)
            dl.text((x0 + 14 * s, y + rh // 2), text, font=f, fill=TEXT, anchor="lm")
        elif kind == "indent":
            dl.text((x0 + 14 * s, y + rh // 2), text, font=f, fill=TEXT, anchor="lm")
        elif kind == "text":
            dl.text((x0, y + rh // 2), text, font=f, fill=TEXT, anchor="lm")
        y += rh
    view = layer.crop((0, scroll, bw, scroll + view_h)) if content_h else layer
    if overflow:   # fade the cut edges
        a = np.asarray(view.getchannel("A"), np.float32)
        n = min(round(18 * s), a.shape[0] // 3)
        if scroll < overflow:
            a[-n:] *= np.linspace(1, 0, n)[:, None]
        if scroll > 0:
            a[:n] *= np.linspace(0, 1, n)[:, None]
        view.putalpha(Image.fromarray(a.astype(np.uint8)))
    img.alpha_composite(view, (box[0], box[1] + round(PAD * s)))
    # the option pills
    y = box[1] + round(PAD * s) + view_h + round(4 * s)
    for line in opt_rows:
        x = box[0] + round(PAD * s)
        for o, i in line:
            w = round(fo.getlength(o) + 24 * s)
            pill = (x, y, x + w, y + round(28 * s))
            d.rounded_rectangle(pill, round(14 * s), fill=(44, 58, 80, 255), outline=EDGE + (150,), width=max(1, round(s)))
            d.text(((pill[0] + pill[2]) / 2, (pill[1] + pill[3]) / 2), o, font=fo, fill=TEXT, anchor="mm")
            regions[f"option:{i}"] = pill
            x += w + round(6 * s)
        y += round(34 * s)
    # the footer: who answered, and open the full app; the close cross in the corner
    fy = box[3] - foot_h // 2 - round(2 * s)
    who = model.get("persona") or ""
    if who and model.get("kind") == "reply":
        d.text((box[0] + round(PAD * s), fy), who.replace("_", " "), font=fs, fill=MUTED, anchor="lm")
    if foot_h:
        open_txt = "Open in app \u2197"
        ow = fs.getlength(open_txt)
        ox = box[2] - round(PAD * s) - ow
        d.text((ox, fy), open_txt, font=fs, fill=LINK, anchor="lm")
        regions["open"] = (ox - 4 * s, fy - 10 * s, box[2] - round(PAD * s) + 4 * s, fy + 10 * s)
    cx, cy, cr = box[2] - round(14 * s), box[1] + round(13 * s), round(4 * s)
    d.line((cx - cr, cy - cr, cx + cr, cy + cr), fill=MUTED, width=max(1, round(1.4 * s)))
    d.line((cx - cr, cy + cr, cx + cr, cy - cr), fill=MUTED, width=max(1, round(1.4 * s)))
    regions["close"] = (cx - 10 * s, cy - 10 * s, cx + 10 * s, cy + 10 * s)
    regions["bubble"] = box
    regions["tail_x"] = tx
    return img, regions, overflow


def frost_backdrop(behind: Image.Image, glass: float = GLASS, scale: float = 1.0) -> Image.Image:
    """What is on the screen behind the chat, frosted: blurred (more, the more glass) and a little more colourful."""
    g = min(100.0, max(0.0, float(glass))) / 100
    blurred = behind.convert("RGB").filter(ImageFilter.GaussianBlur((5 + 15 * g) * scale))
    return ImageEnhance.Color(blurred).enhance(1 + 0.4 * g)


def frost(img: Image.Image, backdrop: Image.Image | None) -> Image.Image:
    """Frosted glass: `backdrop` (frost_backdrop of the screen behind the picture, the same size) under the glass's shape,
    the tint and the text over it. The shadow outside the shape is left as it is. Nothing behind: the picture as it is."""
    shape = img.info.get("shape")
    if backdrop is None or shape is None or backdrop.size != img.size:
        return img
    base = backdrop.convert("RGBA")
    base.putalpha(shape)
    base.alpha_composite(img)   # (outside the shape `base` is clear: the shadow comes through as it was)
    base.info["shape"] = shape
    return base


def premultiplied(img: Image.Image) -> bytes:
    """Straight RGBA -> the premultiplied BGRA bytes UpdateLayeredWindow wants."""
    a = np.asarray(img, np.float32)
    al = a[..., 3:4] / 255
    out = np.dstack([a[..., 2:3] * al, a[..., 1:2] * al, a[..., 0:1] * al, a[..., 3:4]])
    return np.clip(out + 0.5, 0, 255).astype(np.uint8).tobytes()
