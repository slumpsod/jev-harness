"""The floating orb (Windows): a blob of water resting at a screen edge, shown while the app's window is minimized or in
the tray. Click it to chat (the app opens a small chat bubble beside it); drag it anywhere and it settles against the
nearest side of the screen.

A layered window (WS_EX_LAYERED) is given each frame with UpdateLayeredWindow: per-pixel alpha, so the glow blends with
whatever is behind it, and the see-through pixels let clicks through to the windows underneath. (pywebview's
"transparent" windows were measured opaque on Windows 11 / WebView2, 2026-10-12: the glow sat on a grey square.) The
frames come from jevh/orb_art.py, drawn ahead of time. The window lives on its own thread with its own message loop;
other threads talk to it through a queue and a posted message.
"""
from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes
from typing import Callable

from . import orb_art
from .orb_art import settle

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WS_POPUP = 0x80000000
WS_EX_LAYERED, WS_EX_TOPMOST, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x8, 0x80, 0x08000000
WM_DESTROY, WM_TIMER, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE = 0x0002, 0x0113, 0x0201, 0x0202, 0x0200
WM_APP = 0x8000
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA = 2, 0, 1
SWP_NOSIZE, SWP_NOZORDER, SWP_NOACTIVATE = 0x1, 0x4, 0x10
DRAG_PIXELS = 4      # moved less than this between press and release: a click
RADIUS = 34          # of the water, at 100% scaling


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
    user32.MonitorFromWindow.restype = H
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromPoint.restype = H
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [H, ctypes.POINTER(MONITORINFO)]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, wintypes.LPVOID]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
    user32.SetCursor.argtypes = [wintypes.HANDLE]
    user32.SetCapture.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def work_area(x: int, y: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the work area (screen minus taskbar) of the monitor at (x, y)."""
    mon = user32.MonitorFromPoint(wintypes.POINT(x, y), 2)   # MONITOR_DEFAULTTONEAREST
    mi = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    user32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right, r.bottom


class Orb:
    """The orb window. on_click(): it was clicked. on_moved(x, y): it settled somewhere new (save it)."""

    def __init__(self, on_click: Callable[[], None], on_moved: Callable[[int, int], None] | None = None,
                 pos: tuple[int, int] | None = None):
        self.on_click, self.on_moved, self.pos = on_click, on_moved or (lambda x, y: None), pos
        self.hwnd = None
        self.mood = "idle"
        self.visible = False
        self._q: queue.Queue = queue.Queue()
        self._frames: dict[str, list[bytes]] = {}
        self._i = 0
        self._ready = threading.Event()
        self._drag = None          # (cursor x, y, window x, y, moved)
        self._target = None        # where it is settling to
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

    def close(self) -> None:
        self._post("close")

    def rect(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the orb's window, in screen pixels."""
        r = wintypes.RECT()
        if self.hwnd:
            user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        return r.left, r.top, r.right - r.left, r.bottom - r.top

    # -- the window's own thread
    def _run(self) -> None:
        _sigs()
        try:   # physical pixels, and the scaling of the screen it is on
            user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))   # PER_MONITOR_AWARE_V2
            scale = user32.GetDpiForSystem() / 96
        except (AttributeError, OSError):
            scale = 1.0
        self.scale = scale
        radius = max(20, round(RADIUS * scale))
        self.size = orb_art.size_for(radius)
        self._frames["idle"] = orb_art.frames(radius, "idle")
        threading.Thread(target=self._draw_rest, args=(radius,), name="jevh-orb-art", daemon=True).start()
        self._proc = WNDPROC(self._wndproc)   # (kept: a freed callback would crash the process)
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSEXW(cbSize=ctypes.sizeof(WNDCLASSEXW), lpfnWndProc=self._proc, hInstance=hinst,
                         lpszClassName="JevhOrb", hCursor=user32.LoadCursorW(None, ctypes.c_void_p(32649)))  # IDC_HAND
        user32.RegisterClassExW(ctypes.byref(wc))
        w, h = self.size
        if self.pos:
            x, y = self.pos
        else:   # bottom right of the main screen
            left, top, right, bottom = work_area(0, 0)
            x, y = settle(right, bottom - h - 40, w, h, (left, top, right, bottom))
        self.hwnd = user32.CreateWindowExW(WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                                           "JevhOrb", "jev-harness", WS_POPUP, x, y, w, h, None, None, hinst, None)
        self._screen_dc = user32.GetDC(None)
        self._mem_dc = gdi32.CreateCompatibleDC(self._screen_dc)
        bmi = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=w, biHeight=-h, biPlanes=1, biBitCount=32)
        self._bits = ctypes.c_void_p()
        self._bmp = gdi32.CreateDIBSection(self._mem_dc, ctypes.byref(bmi), 0, ctypes.byref(self._bits), None, 0)
        gdi32.SelectObject(self._mem_dc, self._bmp)
        self._paint()
        user32.SetTimer(self.hwnd, 1, 33, None)
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        gdi32.DeleteObject(self._bmp)
        gdi32.DeleteDC(self._mem_dc)
        user32.ReleaseDC(None, self._screen_dc)

    def _draw_rest(self, radius: int) -> None:
        for mood in orb_art.MOODS:
            if mood not in self._frames:
                self._frames[mood] = orb_art.frames(radius, mood)

    def _paint(self) -> None:
        frames = self._frames.get(self.mood) or self._frames["idle"]
        data = frames[self._i % len(frames)]
        ctypes.memmove(self._bits, data, len(data))
        x, y, w, h = self.rect()
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        user32.UpdateLayeredWindow(self.hwnd, self._screen_dc, ctypes.byref(wintypes.POINT(x, y)),
                                   ctypes.byref(wintypes.SIZE(w, h)), self._mem_dc, ctypes.byref(wintypes.POINT(0, 0)), 0,
                                   ctypes.byref(blend), ULW_ALPHA)

    def _move(self, x: int, y: int) -> None:
        user32.SetWindowPos(self.hwnd, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)

    def _commands(self) -> None:
        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "show" and not self.visible:
                self.visible = True
                self._paint()
                user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            elif cmd[0] == "hide" and self.visible:
                self.visible = False
                user32.ShowWindow(self.hwnd, SW_HIDE)
            elif cmd[0] == "mood":
                self.mood = cmd[1]
            elif cmd[0] == "close":
                user32.DestroyWindow(self.hwnd)

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
                if self._target:   # settling against the edge: a quarter of the way each tick
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
                return 0
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
                else:
                    threading.Thread(target=self.on_click, name="jevh-orb-click", daemon=True).start()
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
        except Exception:  # noqa: BLE001 - an exception escaping a window procedure would take the process down
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
