"""The floating orb (Windows): a blob of water resting at a screen edge while the app's window is minimized or in the
tray. Click it: a slim bar slides out to type in; replies come out of the orb as speech bubbles. Drag it anywhere and it
settles against the nearest side of the screen.

Three layered windows (WS_EX_LAYERED), each given its picture with UpdateLayeredWindow: per-pixel alpha, so the glow and
the shadows blend with whatever is behind them, and the see-through pixels let clicks through. (pywebview's "transparent"
windows were measured opaque on Windows 11 / WebView2, 2026-10-12.) The orb's frames come from jevh/orb_art.py, drawn ahead
of time; the bar and the bubble from jevh/orb_chat_art.py, drawn when they change; what they say from jevh/orb_chat.py.
The bar and the bubble are frosted glass: what is behind them on the screen is blurred under their tint. The screen
around the orb is read just before the chat comes out (a copy of the screen includes layered windows, so it cannot be
read with them showing: measured on Windows 11, 2026-10-13), and kept while it is out; how glassy is the Settings tab's "Chat glass" (appearance.orb_glass, 0 to 100),
which the chat page reports with the conversation's state.
The orb and the bubble never take the focus; the bar does, while you type, and a click anywhere else tucks the chat away.
All three live on one thread with its own message loop; other threads talk to it through a queue and a posted message.
"""
from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes
from typing import Callable

from PIL import Image

from . import orb_art, orb_chat_art
from .orb_art import settle
from .orb_chat import ChatState

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WS_POPUP = 0x80000000
WS_EX_LAYERED, WS_EX_TOPMOST, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x8, 0x80, 0x08000000
WM_DESTROY, WM_TIMER, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE = 0x0002, 0x0113, 0x0201, 0x0202, 0x0200
WM_ACTIVATE, WM_CHAR, WM_KEYDOWN, WM_MOUSEWHEEL, WM_MOUSEACTIVATE = 0x0006, 0x0102, 0x0100, 0x020A, 0x0021
WM_APP = 0x8000
MA_NOACTIVATE = 3
SW_HIDE, SW_SHOW, SW_SHOWNOACTIVATE = 0, 5, 4
ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA = 2, 0, 1
SWP_NOSIZE, SWP_NOZORDER, SWP_NOACTIVATE = 0x1, 0x4, 0x10
CF_UNICODETEXT = 13
DRAG_PIXELS = 4      # moved less than this between press and release: a click
RADIUS = 34          # of the water, at 100% scaling
BAR_W = 420          # the bar's width once out, at 100% scaling
GAP = 4              # between the water and the bar


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte), ("SourceConstantAlpha", ctypes.c_byte),
                ("AlphaFormat", ctypes.c_byte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _sigs() -> None:
    H = wintypes.HANDLE
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                       wintypes.HINSTANCE, wintypes.LPVOID]
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
    user32.UpdateLayeredWindow.argtypes = [wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                           ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                           wintypes.COLORREF, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    wintypes.UINT]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.MonitorFromPoint.restype = H
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [H, ctypes.POINTER(MONITORINFO)]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, wintypes.LPVOID]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
    user32.SetCapture.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetFocus.argtypes = [wintypes.HWND]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int,
                             ctypes.c_int, wintypes.DWORD]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE


SRCCOPY = 0x00CC0020


def behind(x: int, y: int, w: int, h: int, screen_dc) -> Image.Image | None:
    """What is on the screen at (x, y, w, h), as RGB: what is behind the chat, for its frosted glass. None when it cannot
    be read (a locked screen, a secure desktop)."""
    if w <= 0 or h <= 0:
        return None
    dc = gdi32.CreateCompatibleDC(screen_dc)
    bits = ctypes.c_void_p()
    bmi = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=w, biHeight=-h, biPlanes=1, biBitCount=32)
    bmp = gdi32.CreateDIBSection(dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    data = None
    try:
        if bmp and bits:
            old = gdi32.SelectObject(dc, bmp)
            if gdi32.BitBlt(dc, 0, 0, w, h, screen_dc, x, y, SRCCOPY):
                data = ctypes.string_at(bits, w * h * 4)
            gdi32.SelectObject(dc, old)
    finally:
        if bmp:
            gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(dc)
    return Image.frombuffer("RGB", (w, h), data, "raw", "BGRX", 0, 1) if data else None


def work_area(x: int, y: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the work area (screen minus taskbar) of the monitor at (x, y)."""
    mon = user32.MonitorFromPoint(wintypes.POINT(x, y), 2)   # MONITOR_DEFAULTTONEAREST
    mi = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    user32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right, r.bottom


class _Placeholder:
    hwnd = None
    visible = False
    pos = (0, 0)
    size = (0, 0)


class _Layer:
    """One layered window: a picture with per-pixel alpha at a place on the screen."""

    def __init__(self, cls: str, hinst, ex_style: int):
        self.hwnd = user32.CreateWindowExW(ex_style, cls, "jev-harness", WS_POPUP, 0, 0, 1, 1, None, None, hinst, None)
        self.size = (0, 0)
        self.pos = (0, 0)
        self.visible = False
        self._dc = gdi32.CreateCompatibleDC(None)
        self._bmp = None
        self._bits = ctypes.c_void_p()

    def put(self, data: bytes, w: int, h: int, x: int, y: int, screen_dc) -> None:
        if (w, h) != self.size:   # a bitmap the size of the picture
            bmi = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=w, biHeight=-h, biPlanes=1, biBitCount=32)
            bmp = gdi32.CreateDIBSection(self._dc, ctypes.byref(bmi), 0, ctypes.byref(self._bits), None, 0)
            gdi32.SelectObject(self._dc, bmp)
            if self._bmp:
                gdi32.DeleteObject(self._bmp)
            self._bmp, self.size = bmp, (w, h)
        ctypes.memmove(self._bits, data, len(data))
        self.pos = (x, y)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        user32.UpdateLayeredWindow(self.hwnd, screen_dc, ctypes.byref(wintypes.POINT(x, y)), ctypes.byref(wintypes.SIZE(w, h)),
                                   self._dc, ctypes.byref(wintypes.POINT(0, 0)), 0, ctypes.byref(blend), ULW_ALPHA)

    def show(self, activate: bool = False) -> None:
        if not self.visible:
            self.visible = True
            user32.ShowWindow(self.hwnd, SW_SHOW if activate else SW_SHOWNOACTIVATE)

    def hide(self) -> None:
        if self.visible:
            self.visible = False
            user32.ShowWindow(self.hwnd, SW_HIDE)


class Orb:
    """The orb window and its chat. on_moved(x, y): it settled somewhere new (save it). on_send(text): a message to the
    assistant. on_answer(question id, text): the answer to its question. on_open_app(): the full app. on_seen(turn id):
    a reply was shown. on_click(): only when the chat is off (chat=False): what a click does instead."""

    def __init__(self, on_click: Callable[[], None] | None = None, on_moved: Callable[[int, int], None] | None = None,
                 pos: tuple[int, int] | None = None, on_send=None, on_answer=None, on_open_app=None, on_seen=None,
                 chat: bool = True):
        self.on_click, self.on_moved, self.pos = on_click or (lambda: None), on_moved or (lambda x, y: None), pos
        self.on_send, self.on_answer = on_send or (lambda t: None), on_answer or (lambda q, t: None)
        self.on_open_app, self.on_seen = on_open_app or (lambda: None), on_seen or (lambda i: None)
        self.chat_on = chat
        self.hwnd = None
        self.mood = "idle"
        self.visible = False
        self.chat = ChatState()
        self.bar_open = False
        self._q: queue.Queue = queue.Queue()
        self._frames: dict[str, list[bytes]] = {}
        self._i = 0
        self._ready = threading.Event()
        self._drag = None          # (cursor x, y, window x, y, moved)
        self._target = None        # where it is settling to
        self._bar_w = 0.0          # the bar's width as it slides out
        self._scroll = 0
        self._regions: dict = {}   # the bubble's clickable parts, in its own pixels
        self._overflow = 0
        self.glass = orb_chat_art.GLASS   # 0 (solid) to 100 (very glassy)
        self._snap = None         # (x, y, the screen around the orb, frosted) from just before the chat came out
        self.size = (0, 0)
        self.scale = 1.0           # the screen's scaling (150% -> 1.5): pywebview's windows are placed unscaled

    # -- from any thread
    def start(self) -> None:
        threading.Thread(target=self._run, name="jevh-orb", daemon=True).start()
        self._ready.wait(10)

    def _post(self, *cmd) -> None:
        self._q.put(cmd)
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_APP, 0, 0)

    def show(self) -> None:
        self._post("show")

    def hide(self) -> None:
        self._post("hide")

    def set_mood(self, mood: str) -> None:
        if mood in orb_art.MOODS and mood != self.mood:
            self._post("mood", mood)

    def chat_state(self, state: dict) -> None:
        """The conversation changed (the app's hidden chat page reports it)."""
        self._post("state", state)

    def close(self) -> None:
        self._post("close")

    def rect(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the orb's window, in screen pixels."""
        r = wintypes.RECT()
        if self.hwnd:
            user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        return r.left, r.top, r.right - r.left, r.bottom - r.top

    # -- the windows' thread
    def _run(self) -> None:
        _sigs()
        try:   # physical pixels, and the scaling of the screen it is on
            user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))   # PER_MONITOR_AWARE_V2
            scale = user32.GetDpiForSystem() / 96
        except (AttributeError, OSError):
            scale = 1.0
        self.scale = scale
        radius = max(20, round(RADIUS * scale))
        self.radius = radius
        self.size = orb_art.size_for(radius)
        self._frames["idle"] = orb_art.frames(radius, "idle")
        threading.Thread(target=self._draw_rest, args=(radius,), name="jevh-orb-art", daemon=True).start()
        self._proc = WNDPROC(self._wndproc)   # (kept: a freed callback would crash the process)
        hinst = kernel32.GetModuleHandleW(None)
        hand = user32.LoadCursorW(None, ctypes.c_void_p(32649))   # IDC_HAND
        arrow = user32.LoadCursorW(None, ctypes.c_void_p(32512))  # IDC_ARROW
        for cls, cur in (("JevhOrb", hand), ("JevhOrbBar", arrow), ("JevhOrbBubble", arrow)):
            wc = WNDCLASSEXW(cbSize=ctypes.sizeof(WNDCLASSEXW), lpfnWndProc=self._proc, hInstance=hinst,
                             lpszClassName=cls, hCursor=cur)
            user32.RegisterClassExW(ctypes.byref(wc))
        quiet = WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        self._screen_dc = user32.GetDC(None)
        # (placeholders first: the window procedure runs while each window is being made, before the others exist)
        self._orb = self._bar = self._bubble = _Placeholder()
        self._orb = _Layer("JevhOrb", hinst, quiet)
        self._bar = _Layer("JevhOrbBar", hinst, WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW)   # (it takes the keys)
        self._bubble = _Layer("JevhOrbBubble", hinst, quiet)
        self.hwnd = self._orb.hwnd
        w, h = self.size
        if self.pos:
            x, y = self.pos
        else:   # bottom right of the main screen
            left, top, right, bottom = work_area(0, 0)
            x, y = settle(right, bottom - h - 40, w, h, (left, top, right, bottom))
        self._orb.pos = (x, y)
        self._paint()
        user32.SetTimer(self.hwnd, 1, 33, None)
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.ReleaseDC(None, self._screen_dc)

    def _draw_rest(self, radius: int) -> None:
        for mood in orb_art.MOODS:
            if mood not in self._frames:
                self._frames[mood] = orb_art.frames(radius, mood)

    def _paint(self) -> None:
        frames = self._frames.get(self.mood) or self._frames["idle"]
        w, h = self.size
        x, y = self._orb.pos
        self._orb.put(frames[self._i % len(frames)], w, h, x, y, self._screen_dc)

    def _move(self, x: int, y: int) -> None:
        self._orb.pos = (x, y)
        user32.SetWindowPos(self.hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)

    # -- the chat
    def _water(self) -> tuple[float, float]:
        """Where the water's middle is on the screen (it rests a little below the middle of its frame)."""
        x, y = self._orb.pos
        w, h = self.size
        return x + w / 2, y + h / 2 + 0.08 * self.radius

    def _side(self) -> str:
        """"right": the orb is on the right half of its screen, so the chat opens to its left."""
        cx, cy = self._water()
        left, _, right, _ = work_area(int(cx), int(cy))
        return "right" if cx > (left + right) / 2 else "left"

    def _snapshot(self) -> None:
        """Read the screen around the orb, where the bar and the bubble can go, and frost it. Only while neither shows."""
        self._snap = None
        if self.glass <= 2:
            return
        s = self.scale
        cx, cy = self._water()
        left, top, right, bottom = work_area(int(cx), int(cy))
        span_x, span_y = (BAR_W + 80) * s + self.radius, (orb_chat_art.BUBBLE_MAX_H + 160) * s
        x0, y0 = int(max(left, cx - span_x)), int(max(top, cy - span_y))
        x1, y1 = int(min(right, cx + span_x)), int(min(bottom, cy + span_y))
        shot = behind(x0, y0, x1 - x0, y1 - y0, self._screen_dc)
        if shot is not None:
            self._snap = (x0, y0, orb_chat_art.frost_backdrop(shot, self.glass, s))

    def _frosted(self, img, x: int, y: int):
        """`img` as frosted glass over what was behind it at (x, y) when the chat came out."""
        if self.glass <= 2 or not self._snap:
            return img
        x0, y0, snap = self._snap
        bx, by = x - x0, y - y0
        if bx < 0 or by < 0 or bx + img.width > snap.width or by + img.height > snap.height:
            return img
        return orb_chat_art.frost(img, snap.crop((bx, by, bx + img.width, by + img.height)))

    def _draw_bar(self) -> None:
        s = self.scale
        img, _ = orb_chat_art.draw_bar(self.chat.text, max(1, round(self._bar_w)), s, caret=(self._i // 16) % 2 == 0,
                                       caret_at=self.chat.caret, secret=self.chat.secret,
                                       placeholder="Type it (hidden)" if self.chat.secret else
                                       "Answer, or pick above" if self.chat.question else "Ask anything...",
                                       busy=self.chat.awaiting is not None and self.chat.busy, glass=self.glass)
        cx, cy = self._water()
        m = round(10 * s)
        x = (cx - self.radius - GAP * s - img.width + m) if self._side() == "right" else (cx + self.radius + GAP * s - m)
        x, y = round(x), round(cy - img.height / 2)
        img = self._frosted(img, x, y)
        self._bar.put(orb_chat_art.premultiplied(img), img.width, img.height, x, y, self._screen_dc)

    def _draw_bubble(self) -> None:
        if not (self.chat.bubble and self.chat.bubble_open):
            self._bubble.hide()
            return
        if not self._bubble.visible and not self._bar.visible:   # coming out on its own: what is behind it first
            self._snapshot()
        s = self.scale
        side = self._side()
        cx, cy = self._water()
        left, top, right, bottom = work_area(int(cx), int(cy))
        img, regions, overflow = orb_chat_art.draw_bubble(self.chat.bubble, s, tail=side, scroll=self._scroll,
                                                          tick=self._i // 12, glass=self.glass)
        above = cy - self.radius - (img.height if not self.bar_open else img.height + 0) > top
        if not above:
            img, regions, overflow = orb_chat_art.draw_bubble(self.chat.bubble, s, tail=side, below=True,
                                                              scroll=self._scroll, tick=self._i // 12, glass=self.glass)
        self._regions, self._overflow = regions, overflow
        # the tail's tip points at the water: just above it (or below), a little in from its middle
        tip = regions["tail_x"]
        tx = cx - 0.15 * self.radius if side == "right" else cx + 0.15 * self.radius
        x = tx - tip - (8 if side == "right" else -8) * s
        if self.bar_open:   # with the bar out: the bubble sits over the bar, its tail toward the orb
            bx = self._bar.pos[0]
            m = round(14 * s)
            x = (bx + self._bar.size[0] - img.width + m) if side == "right" else bx - m
            y = (self._bar.pos[1] - img.height + round(16 * s)) if above else (self._bar.pos[1] + self._bar.size[1] - round(16 * s))
        else:
            y = (cy - self.radius * 0.9 - img.height) if above else (cy + self.radius * 0.9)
        x, y = round(min(max(x, left), right - img.width)), round(y)
        img = self._frosted(img, x, y)
        self._bubble.put(orb_chat_art.premultiplied(img), img.width, img.height, x, y, self._screen_dc)
        self._bubble.show()
        if self.chat.bubble.get("kind") == "reply" and self.chat.bubble.get("id"):
            self.on_seen(self.chat.bubble["id"])

    def _open_bar(self) -> None:
        self._bubble.hide()   # (the screen behind is read with neither showing; the bubble comes back below)
        self._snapshot()
        self.bar_open, self._bar_w = True, float(round(40 * self.scale))
        self._draw_bar()
        self._bar.show(activate=True)
        user32.SetForegroundWindow(self._bar.hwnd)
        user32.SetFocus(self._bar.hwnd)
        if self.chat.bubble:
            self.chat.bubble_open = True
            self._draw_bubble()

    def _close_chat(self) -> None:
        self.bar_open = False
        self._bar.hide()
        self.chat.dismiss()
        self._bubble.hide()

    def _paste(self) -> None:
        if not user32.OpenClipboard(self._bar.hwnd):
            return
        try:
            h = user32.GetClipboardData(CF_UNICODETEXT)
            if h:
                p = kernel32.GlobalLock(h)
                if p:
                    self.chat.insert(ctypes.wstring_at(p))
                    kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()

    def _act(self, action) -> None:
        if not action:
            return
        if action[0] == "send":
            self.chat.sent(action[1])
            self._scroll = 0
            threading.Thread(target=self.on_send, args=(action[1],), name="jevh-orb-send", daemon=True).start()
        elif action[0] == "answer":
            self.chat.answered()
            threading.Thread(target=self.on_answer, args=(action[1], action[2]), name="jevh-orb-answer", daemon=True).start()
        elif action[0] == "close":
            self._close_chat()
        elif action[0] == "paste":
            self._paste()
        if self.bar_open:
            self._draw_bar()
        self._draw_bubble()

    def _click_bubble(self, x: int, y: int) -> None:
        def inside(r):
            return r and r[0] <= x <= r[2] and r[1] <= y <= r[3]
        if inside(self._regions.get("close")):
            self.chat.dismiss()
            self._bubble.hide()
            return
        if inside(self._regions.get("open")):
            self._close_chat()
            threading.Thread(target=self.on_open_app, name="jevh-orb-open", daemon=True).start()
            return
        for k, r in self._regions.items():
            if k.startswith("option:") and inside(r) and self.chat.question:
                opts = self.chat.question.get("options") or []
                i = int(k.split(":")[1])
                if i < len(opts):
                    self._act(("answer", self.chat.question["id"], opts[i]))
                return

    def _commands(self) -> None:
        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "show" and not self.visible:
                self.visible = True
                self._paint()
                self._orb.show()
                if self.chat.bubble and self.chat.bubble_open:
                    self._draw_bubble()
            elif cmd[0] == "hide" and self.visible:
                self.visible = False
                self._orb.hide()
                self._close_chat()
            elif cmd[0] == "mood":
                self.mood = cmd[1]
            elif cmd[0] == "state":
                pop = self.chat.update(cmd[1])
                glass = cmd[1].get("glass")
                if isinstance(glass, (int, float)) and not isinstance(glass, bool) and float(glass) != self.glass:
                    self.glass = min(100.0, max(0.0, float(glass)))
                    self._snap = None   # (read again, at the new level, the next time the chat comes out)
                if self.visible and (pop or self.bar_open or (self.chat.bubble_open and self._bubble.visible)):
                    if self.bar_open:
                        self._draw_bar()
                    self._draw_bubble()
            elif cmd[0] == "close":
                for layer in (self._bubble, self._bar, self._orb):
                    user32.DestroyWindow(layer.hwnd)

    def _cursor(self) -> tuple[int, int]:
        p = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(p))
        return p.x, p.y

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_APP:
                self._commands()
                return 0
            if msg == WM_TIMER:
                self._commands()
                if self._target:   # settling against the edge: a third of the way each tick
                    x, y, _, _ = self.rect()
                    tx, ty = self._target
                    nx, ny = x + (tx - x) * 0.3, y + (ty - y) * 0.3
                    if abs(tx - nx) < 1.5 and abs(ty - ny) < 1.5:
                        nx, ny, self._target = tx, ty, None
                        self.on_moved(int(tx), int(ty))
                    self._move(int(round(nx)), int(round(ny)))
                if self.visible:
                    self._i += 1
                    self._paint()
                    if self.bar_open:
                        full = BAR_W * self.scale
                        if self._bar_w < full:   # sliding out
                            self._bar_w = min(full, self._bar_w + (full - self._bar_w) * 0.35 + 6)
                            self._draw_bar()
                            if self.chat.bubble_open:
                                self._draw_bubble()
                        elif self._i % 16 == 0:   # the caret blinks
                            self._draw_bar()
                    if self._bubble.visible and self.chat.bubble and self.chat.bubble.get("kind") == "waiting" \
                            and self._i % 12 == 0:
                        self._draw_bubble()
                return 0
            if hwnd == self._bubble.hwnd:
                if msg == WM_MOUSEACTIVATE:
                    return MA_NOACTIVATE   # (a click on the bubble keeps the focus in the bar)
                if msg == WM_LBUTTONUP:
                    self._click_bubble(ctypes.c_short(lparam & 0xFFFF).value, ctypes.c_short((lparam >> 16) & 0xFFFF).value)
                    return 0
                if msg == WM_MOUSEWHEEL and self._overflow:
                    delta = ctypes.c_short((wparam >> 16) & 0xFFFF).value
                    self._scroll = max(0, min(self._overflow, self._scroll - round(delta / 120 * 48 * self.scale)))
                    self._draw_bubble()
                    return 0
            if hwnd == self._bar.hwnd:
                if msg == WM_CHAR:
                    self._act(self.chat.char(int(wparam)))
                    if self.bar_open:
                        self._draw_bar()
                    return 0
                if msg == WM_KEYDOWN and wparam in (0x25, 0x27, 0x24, 0x23, 0x2E):
                    self.chat.key(int(wparam))
                    self._draw_bar()
                    return 0
                if msg == WM_LBUTTONUP:
                    x = ctypes.c_short(lparam & 0xFFFF).value
                    send = orb_chat_art.draw_bar(self.chat.text, max(1, round(self._bar_w)), self.scale)[1].get("send")
                    if send and send[0] <= x <= send[2]:
                        self._act(self.chat.char(13))
                    return 0
                if msg == WM_ACTIVATE and (wparam & 0xFFFF) == 0 and self.bar_open:
                    self._close_chat()   # a click anywhere else: the chat tucks itself away
                    return 0
            if hwnd == self.hwnd:   # the orb
                if msg == WM_LBUTTONDOWN:
                    cx, cy = self._cursor()
                    x, y, _, _ = self.rect()
                    self._drag = [cx, cy, x, y, False]
                    self._target = None
                    user32.SetCapture(hwnd)
                    return 0
                if msg == WM_MOUSEMOVE and self._drag:
                    cx, cy = self._cursor()
                    d = self._drag
                    if not d[4] and (abs(cx - d[0]) > DRAG_PIXELS or abs(cy - d[1]) > DRAG_PIXELS):
                        d[4] = True
                        if self.bar_open or self._bubble.visible:
                            self._close_chat()
                    if d[4]:
                        self._move(d[2] + cx - d[0], d[3] + cy - d[1])
                    return 0
                if msg == WM_LBUTTONUP and self._drag:
                    user32.ReleaseCapture()
                    dragged = self._drag[4]
                    self._drag = None
                    if dragged:
                        x, y, w, h = self.rect()
                        self._target = settle(x, y, w, h, work_area(x + w // 2, y + h // 2))
                    elif not self.chat_on:
                        threading.Thread(target=self.on_click, name="jevh-orb-click", daemon=True).start()
                    elif self.bar_open:
                        self._close_chat()
                    else:
                        self._open_bar()
                    return 0
            if msg == WM_DESTROY and hwnd == self.hwnd:
                user32.PostQuitMessage(0)
                return 0
        except Exception:  # noqa: BLE001 - an exception escaping a window procedure would take the process down;
            pass               # (and returning 0 instead would refuse WM_NCCREATE: the window would never be made)
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
