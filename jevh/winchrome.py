"""The app window's own title bar (Windows): no Windows frame, the page's top strip drags it, and it keeps what a framed
window has - resizing at the edges, Aero Snap, double-click to maximize, the minimize/maximize animations, the shadow.

How: the page marks its strip `app-region: drag` and WebView2 (IsNonClientRegionSupportEnabled) lets those clicks through
to the window, which answers WM_NCHITTEST with HTCAPTION there and HTLEFT/HTTOP/... on a thin border around the page. The
window keeps WS_THICKFRAME/WS_CAPTION (so Windows snaps and animates it) and answers WM_NCCALCSIZE so that no frame is
drawn. The window procedure is subclassed from ctypes, on top of WinForms' own.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
GWL_STYLE, GWLP_WNDPROC = -16, -4
WS_CAPTION, WS_THICKFRAME, WS_MINIMIZEBOX, WS_MAXIMIZEBOX, WS_SYSMENU = 0xC00000, 0x40000, 0x20000, 0x10000, 0x80000
WM_NCCALCSIZE, WM_NCHITTEST, WM_SYSCOMMAND = 0x0083, 0x0084, 0x0112
HTCLIENT, HTCAPTION = 1, 2
HTLEFT, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT, HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT = 10, 11, 12, 13, 14, 15, 16, 17
SC_MINIMIZE, SC_MAXIMIZE, SC_RESTORE, SC_CLOSE = 0xF020, 0xF030, 0xF120, 0xF060
SWP_FRAMECHANGED, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SWP_NOACTIVATE = 0x20, 0x2, 0x1, 0x4, 0x10


class NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
user32.CallWindowProcW.restype = LRESULT
user32.CallWindowProcW.argtypes = [ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                wintypes.UINT]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.IsZoomed.argtypes = [wintypes.HWND]
user32.MonitorFromWindow.restype = wintypes.HANDLE
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.GetDpiForWindow.argtypes = [wintypes.HWND]


class Chrome:
    """Installed on the app's main window (its HWND) once it exists; `grip` is the resize border, in unscaled pixels."""

    def __init__(self, hwnd: int, grip: int = 5):
        self.hwnd, self.grip = hwnd, grip
        self._old = 0
        self._proc = WNDPROC(self._wndproc)   # (kept: a freed callback would crash the process)

    def install(self) -> None:
        style = user32.GetWindowLongPtrW(self.hwnd, GWL_STYLE)
        user32.SetWindowLongPtrW(self.hwnd, GWL_STYLE,
                                 style | WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
        self._old = user32.SetWindowLongPtrW(self.hwnd, GWLP_WNDPROC, ctypes.cast(self._proc, ctypes.c_void_p).value)
        user32.SetWindowPos(self.hwnd, None, 0, 0, 0, 0,
                            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)

    # what the page's buttons do (any thread: posted to the window's own)
    def minimize(self) -> None:
        user32.PostMessageW(self.hwnd, WM_SYSCOMMAND, SC_MINIMIZE, 0)

    def toggle_maximize(self) -> None:
        user32.PostMessageW(self.hwnd, WM_SYSCOMMAND, SC_RESTORE if self.maximized() else SC_MAXIMIZE, 0)

    def maximized(self) -> bool:
        return bool(user32.IsZoomed(self.hwnd))

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_NCCALCSIZE and wparam:
                # no frame: the client is the whole window - except when maximized, where Windows puts the window's
                # invisible frame off-screen, so the client is kept to the screen's work area (or the page would be cut)
                if user32.IsZoomed(hwnd):
                    p = ctypes.cast(lparam, ctypes.POINTER(NCCALCSIZE_PARAMS)).contents
                    mi = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
                    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 2), ctypes.byref(mi))
                    p.rgrc[0] = mi.rcWork
                return 0
            if msg == WM_NCHITTEST:
                x, y = ctypes.c_short(lparam & 0xFFFF).value, ctypes.c_short((lparam >> 16) & 0xFFFF).value
                r = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(r))
                g = max(2, round(self.grip * user32.GetDpiForWindow(hwnd) / 96))
                if not user32.IsZoomed(hwnd):
                    left, right = x < r.left + g, x >= r.right - g
                    top, bottom = y < r.top + g, y >= r.bottom - g
                    if top and left: return HTTOPLEFT       # noqa: E701
                    if top and right: return HTTOPRIGHT     # noqa: E701
                    if bottom and left: return HTBOTTOMLEFT  # noqa: E701
                    if bottom and right: return HTBOTTOMRIGHT  # noqa: E701
                    if left: return HTLEFT                  # noqa: E701
                    if right: return HTRIGHT                # noqa: E701
                    if top: return HTTOP                    # noqa: E701
                    if bottom: return HTBOTTOM              # noqa: E701
                # anything else that reaches this window rather than the page is the page's drag strip
                # (WebView2 passes `app-region: drag` through): the title bar
                return HTCAPTION
        except Exception:  # noqa: BLE001 - never let an exception out of a window procedure
            pass
        return user32.CallWindowProcW(self._old, hwnd, msg, wparam, lparam)
