"""The orb's pictures: a blob of water rolling gently back and forth, with a glow, as premultiplied BGRA frames.

Drawn ahead of time (numpy + Pillow's blur), a loop per mood, so the window that shows them (jevh/orb.py) only hands one
frame a tick to Windows. A frame's transparent pixels are click-through there, so only the water and its glow catch clicks.

Moods: idle (cyan glow), working (brighter glow, the water churns faster), attention (an amber pulse: the assistant is
asking you something), unread (a cyan breathing pulse: an answer you haven't read).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

FPS = 30
LOOP = 4.0          # seconds a loop lasts (every motion below repeats a whole number of times in it: no seam)
MOODS = ("idle", "working", "attention", "unread")
MARGIN = 12          # from the screen edge the orb settles at (jevh/orb.py)
GLOW = {"idle": (80, 200, 255), "working": (110, 225, 255), "attention": (255, 170, 60), "unread": (90, 210, 255)}


def size_for(radius: int) -> tuple[int, int]:
    """(width, height) of a frame: room for the roll (+-0.35 R) and the glow around it."""
    return int(radius * 3.6) // 2 * 2, int(radius * 3.0) // 2 * 2


def frame(t: float, radius: int = 34, mood: str = "idle") -> np.ndarray:
    """One frame at loop time t in [0, 1): an (h, w, 4) uint8 array, premultiplied BGRA."""
    R = float(radius)
    w, h = size_for(radius)
    tau = 2 * math.pi
    fast = 2 if mood == "working" else 1
    # the roll: a gentle back and forth along the floor, the inside turning with it as a rolling ball's would
    dx = 0.35 * R * math.sin(tau * t)
    v = math.cos(tau * t)                      # its speed (-1..1): the drop stretches a little while it moves
    phi = dx / R
    cx, cy = w / 2 + dx, h / 2 + 0.08 * R
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xs - cx, ys - cy
    sx = 1 + 0.06 * abs(v)
    xe, ye = x / sx, y * sx
    rho = np.hypot(xe, ye)
    th = np.arctan2(ye, xe)
    # the wobble: a few slow waves round the rim, and a slightly flattened bottom where it rests
    k = tau * t * fast
    r = R * (1 + 0.020 * np.sin(2 * th + 2 * k) + 0.016 * np.sin(3 * th - 3 * k) + 0.008 * np.sin(5 * th + k)
             - 0.06 * np.clip(np.sin(th), 0, None) ** 3)
    inside = np.clip((r - rho) / 1.3 + 0.5, 0, 1)           # antialiased edge
    n = np.clip(rho / r, 0, 1)                               # 0 at the middle, 1 at the rim
    # water: deep at the rim, clear in the middle (refraction), a bright fresnel rim
    deep, clear = np.array([190, 92, 12], np.float32), np.array([255, 205, 110], np.float32)   # (BGR)
    mix = (n ** 1.6)[..., None]
    body = clear * (1 - mix) + deep * mix
    rim = np.clip((n - 0.78) / 0.22, 0, 1) ** 2
    body += rim[..., None] * np.array([255, 245, 190], np.float32) * 0.45
    # caustics: soft bands that turn as it rolls and drift (faster while working)
    u = (x * math.cos(phi) - y * math.sin(phi)) / R
    q = (x * math.sin(phi) + y * math.cos(phi)) / R
    caus = 0.5 + 0.5 * np.sin(5.5 * u + 2.5 * np.sin(3.5 * q + 2 * k) + k)
    body += (caus ** 3 * np.clip(1 - n, 0, 1) * 70)[..., None]
    # the glints: fixed to the light (upper left), not to the turning water
    a1, b1, rot = 0.30 * R, 0.17 * R, math.radians(-32)
    hx, hy = x + 0.33 * R, y + 0.40 * R
    gu, gv = hx * math.cos(rot) - hy * math.sin(rot), hx * math.sin(rot) + hy * math.cos(rot)
    spec = np.exp(-((gu / a1) ** 2 + (gv / b1) ** 2) * 2.2)
    spec2 = np.exp(-(((x - 0.30 * R) / (0.13 * R)) ** 2 + ((y - 0.40 * R) / (0.07 * R)) ** 2) * 2.0) * 0.35
    body += ((spec + spec2) * 255)[..., None]
    body = np.clip(body, 0, 255)
    body_a = inside * (0.66 + 0.31 * n)                      # glassy: the desktop shows through the middle, less at the rim
    # the glow: the shape blurred, in the mood's colour and strength
    strength = {"idle": 0.42, "working": 0.75,
                "attention": 0.45 + 0.4 * (0.5 + 0.5 * math.sin(tau * t * 4)),
                "unread": 0.30 + 0.35 * (0.5 + 0.5 * math.sin(tau * t * 2))}[mood]
    mask = Image.fromarray((inside * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(R * 0.38))
    glow_a = np.asarray(mask, np.float32) / 255 * strength
    gb, gg, gr = GLOW[mood][2], GLOW[mood][1], GLOW[mood][0]
    glow = np.array([gb, gg, gr], np.float32)
    # premultiplied "water over glow"
    out_a = body_a + glow_a * (1 - body_a)
    out_rgb = body * body_a[..., None] + glow * (glow_a * (1 - body_a))[..., None]
    out = np.dstack([out_rgb, out_a[..., None] * 255])
    return np.clip(out, 0, 255).astype(np.uint8)


def settle(x: int, y: int, w: int, h: int, area: tuple[int, int, int, int]) -> tuple[int, int]:
    """Where a window at (x, y), w x h, settles: against the nearer left or right edge of `area`, inside it top to bottom."""
    left, top, right, bottom = area
    cx = x + w / 2
    nx = left + MARGIN - w // 6 if cx - left < right - cx else right - w + w // 6 - MARGIN   # (the frame has room around the water)
    ny = min(max(y, top), bottom - h)
    return int(nx), int(ny)


def frames(radius: int = 34, mood: str = "idle") -> list[bytes]:
    """A whole loop for one mood: LOOP * FPS frames, each the raw BGRA bytes of a frame()."""
    count = int(LOOP * FPS)
    return [frame(i / count, radius, mood).tobytes() for i in range(count)]


def preview(path: str, radius: int = 34, mood: str = "idle", t: float = 0.1, backdrop=(30, 34, 40)) -> None:
    """A frame over a plain backdrop, saved as a PNG (to look at it without Windows)."""
    f = frame(t, radius, mood).astype(np.float32)
    a = f[..., 3:4] / 255
    bg = np.array(backdrop[::-1], np.float32)
    rgb = f[..., :3] + bg * (1 - a)                          # premultiplied over: src + dst * (1 - a)
    Image.fromarray(np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8), "RGB").save(path)
