"""Windows plumbing for the `desktop` tool: the window list, UI Automation snapshots and patterns, real mouse and
keyboard input, the clipboard and screenshots. Windows only; everything is imported on first use, so importing this
module on another system is harmless and the tests drive `desktop` through a fake with the same methods.

Why a layer of its own: in the live session that led to it (20261008-093548-5045) the worker model spent 37 steps
getting a window list out of raw `comtypes` calls (a property condition with the wrong variant, an interface that
would not import, a tree walk that took minutes), re-did the same setup on every later turn and typed with
`keybd_event`. Here the calls are made once, correctly: one cached UIA query returns a whole window tree in tens of
milliseconds, text goes in by real input events (or the clipboard when it is long or has line breaks), and keys are only
sent to a window that is in front.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from ctypes import wintypes as wt

IS_WINDOWS = sys.platform == "win32"

# UIA property ids (UIAutomationClient.h)
P_RECT, P_PID, P_CTYPE, P_NAME, P_FOCUSABLE, P_ENABLED, P_AID, P_CLASS = 30001, 30002, 30003, 30005, 30009, 30010, 30011, 30012
P_HELP, P_PASSWORD, P_OFFSCREEN, P_HASFOCUS = 30013, 30019, 30022, 30008
P_VALUE, P_READONLY, P_TOGGLE, P_SELECTED, P_EXPAND = 30045, 30046, 30086, 30079, 30070
AVAIL = {"invoke": 30031, "value": 30043, "toggle": 30041, "select": 30036, "expand": 30028, "scroll": 30034, "text": 30040}
PATTERN = {"invoke": 10000, "value": 10002, "scroll": 10004, "expand": 10005, "select": 10010, "toggle": 10015,
           "text": 10014, "window": 10009}

CONTROL_TYPES = {50000: "Button", 50001: "Calendar", 50002: "CheckBox", 50003: "ComboBox", 50004: "Edit", 50005: "Hyperlink",
                 50006: "Image", 50007: "ListItem", 50008: "List", 50009: "Menu", 50010: "MenuBar", 50011: "MenuItem",
                 50012: "ProgressBar", 50013: "RadioButton", 50014: "ScrollBar", 50015: "Slider", 50016: "Spinner",
                 50017: "StatusBar", 50018: "Tab", 50019: "TabItem", 50020: "Text", 50021: "ToolBar", 50022: "ToolTip",
                 50023: "Tree", 50024: "TreeItem", 50025: "Custom", 50026: "Group", 50027: "Thumb", 50028: "DataGrid",
                 50029: "DataItem", 50030: "Document", 50031: "SplitButton", 50032: "Window", 50033: "Pane",
                 50034: "Header", 50035: "HeaderItem", 50036: "Table", 50037: "TitleBar", 50038: "Separator"}

VK = {"enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09, "space": 0x20, "backspace": 0x08,
      "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
      "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12,
      "win": 0x5B, "windows": 0x5B, "meta": 0x5B, "cmd": 0x5B, "menu": 0x5D, "capslock": 0x14, "printscreen": 0x2C}
VK.update({f"f{n}": 0x6F + n for n in range(1, 25)})
EXTENDED = {0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22, 0x25, 0x26, 0x27, 0x28, 0x5B, 0x5D}
MODIFIERS = {"ctrl", "control", "shift", "alt", "win", "windows", "meta", "cmd"}

KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_EXTENDEDKEY = 0x0002, 0x0004, 0x0001
MOUSEEVENTF = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
MOUSEEVENTF_WHEEL = 0x0800


def parse_combo(combo: str) -> list[int]:
    """'ctrl+shift+n' -> virtual-key codes, modifiers first. Raises ValueError for a key it does not know."""
    parts = [p.strip().lower() for p in re.split(r"\s*\+\s*(?=.)", combo.strip()) if p.strip()]
    if not parts:
        raise ValueError("no key given")
    mods = [VK[p] for p in parts[:-1] if p in MODIFIERS]
    if len(mods) != len(parts) - 1:
        raise ValueError(f"only the last part of {combo!r} can be a plain key")
    last = parts[-1]
    if last in VK:
        code = VK[last]
    elif len(last) == 1 and last.isascii():
        if IS_WINDOWS:
            scan = ctypes.windll.user32.VkKeyScanW(ord(last)) & 0xFF
            code = scan if scan != 0xFF else ord(last.upper())
        else:
            code = ord(last.upper())
    else:
        raise ValueError(f"unknown key {last!r}")
    return mods + [code]


if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi")

    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                    ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                    ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _INPUTU(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wt.DWORD), ("u", _INPUTU)]

    user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wt.UINT
    user32.GetForegroundWindow.restype = wt.HWND
    user32.SwitchToThisWindow.argtypes = [wt.HWND, wt.BOOL]
    user32.WindowFromPoint.argtypes = [wt.POINT]
    user32.WindowFromPoint.restype = wt.HWND
    user32.GetWindow.restype = wt.HWND
    user32.GetWindow.argtypes = [wt.HWND, wt.UINT]
    user32.GetAncestor.restype = wt.HWND
    user32.GetAncestor.argtypes = [wt.HWND, wt.UINT]
    user32.OpenClipboard.argtypes = [wt.HWND]
    user32.GetClipboardData.argtypes = [wt.UINT]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [wt.UINT, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [wt.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def _need_windows() -> None:
    if not IS_WINDOWS:
        raise RuntimeError("the desktop tool drives Windows (UI Automation); this is not Windows")


class WinUI:
    """The real backend. Every method may be called from one thread only (the tool's own thread: COM objects
    belong to the apartment that made them)."""

    def __init__(self) -> None:
        _need_windows()
        try:   # UIA rectangles are physical pixels; make our mouse coordinates the same
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:  # noqa: BLE001
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:  # noqa: BLE001
                pass
        import comtypes
        import comtypes.client
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as U
        self.U, self.comtypes = U, comtypes
        try:   # Windows 10+: a frozen app must not freeze us (an unanswered UIA call otherwise waits 20 s, per call)
            self.iuia = comtypes.CoCreateInstance(U.CUIAutomation8._reg_clsid_, interface=U.IUIAutomation6,
                                                  clsctx=comtypes.CLSCTX_INPROC_SERVER)
            self.iuia.ConnectionTimeout, self.iuia.TransactionTimeout = 2000, 5000
        except Exception:  # noqa: BLE001
            self.iuia = comtypes.CoCreateInstance(U.CUIAutomation._reg_clsid_, interface=U.IUIAutomation,
                                                  clsctx=comtypes.CLSCTX_INPROC_SERVER)
        self.cache = self.iuia.CreateCacheRequest()
        for p in (P_RECT, P_PID, P_CTYPE, P_NAME, P_ENABLED, P_AID, P_CLASS, P_HELP, P_PASSWORD, P_OFFSCREEN,
                  P_HASFOCUS, P_VALUE, P_READONLY, P_TOGGLE, P_SELECTED, P_EXPAND, *AVAIL.values()):
            self.cache.AddProperty(p)
        self.tree = self.iuia.ControlViewCondition
        self.pid_exe: dict[int, str] = {}

    # ------------------------------------------------------------------ windows
    def _exe(self, pid: int) -> str:
        if pid in self.pid_exe:
            return self.pid_exe[pid]
        name = ""
        h = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            buf, size = ctypes.create_unicode_buffer(520), wt.DWORD(520)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                name = os.path.splitext(os.path.basename(buf.value))[0]
            kernel32.CloseHandle(h)
        self.pid_exe[pid] = name
        return name

    def windows(self) -> list[dict]:
        """Every visible, uncloaked window with a size, top of the z-order first. Owned windows (dialogs, menus) are
        included with their `owner`; the caller decides what is a window of its own."""
        out: list[dict] = []
        fg = user32.GetForegroundWindow()

        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            cloaked = wt.DWORD(0)
            dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), 4)
            if cloaked.value:
                return True
            rect = wt.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, title, 512)
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            pid = wt.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            ex = user32.GetWindowLongW(hwnd, -20)
            owner = user32.GetWindow(hwnd, 4) or 0   # GW_OWNER
            minimized = bool(user32.IsIconic(hwnd))
            if not minimized and (rect.right - rect.left < 2 or rect.bottom - rect.top < 2):
                return True
            out.append({"hwnd": int(hwnd), "pid": pid.value, "exe": self._exe(pid.value), "title": title.value,
                        "cls": cls.value, "rect": (rect.left, rect.top, rect.right, rect.bottom),
                        "minimized": minimized, "foreground": bool(hwnd == fg), "owner": int(owner),
                        "tool": bool(ex & 0x80) and not (ex & 0x40000), "enabled": bool(user32.IsWindowEnabled(hwnd))})
            return True

        user32.EnumWindows(WNDENUMPROC(cb), 0)
        return out

    def foreground(self) -> int:
        return int(user32.GetForegroundWindow() or 0)

    def in_front(self, hwnd: int) -> bool:
        """Is `hwnd` (or a window it owns, like its dialog) the one that gets the keys?"""
        fg = user32.GetForegroundWindow()
        if not fg:
            return False
        if int(fg) == hwnd or int(user32.GetAncestor(fg, 3) or 0) == hwnd:   # GA_ROOTOWNER
            return True
        a, b = wt.DWORD(0), wt.DWORD(0)
        user32.GetWindowThreadProcessId(fg, ctypes.byref(a))
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(b))
        return a.value == b.value and self._exe(a.value) not in ("ApplicationFrameHost", "explorer")

    def focus(self, hwnd: int) -> bool:
        """Bring a window to the front. Windows refuses SetForegroundWindow from a process that did not just get input, and
        sometimes a window nobody can see holds the foreground (live, 2026-10-08: a service's hidden window took it when the
        front app closed, and five different API attempts, the worker's own included, could not take it back). So the ladder
        ends in what the lock does not apply to: a real click on the target's own title area."""
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE
        for attempt in range(5):
            if attempt == 1:   # Windows only lets the foreground process hand focus on; a tap on Alt counts as input
                user32.keybd_event(0x12, 0, 0, 0)
                user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0)
            elif attempt == 2:
                fg = user32.GetForegroundWindow()
                me = kernel32.GetCurrentThreadId()
                other = user32.GetWindowThreadProcessId(fg, None) if fg else 0
                if other:
                    user32.AttachThreadInput(me, other, True)
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                if other:
                    user32.AttachThreadInput(me, other, False)
            elif attempt == 3:
                user32.SwitchToThisWindow(hwnd, True)
            elif attempt == 4:
                if not self._click_to_activate(hwnd):
                    user32.ShowWindow(hwnd, 6)   # SW_MINIMIZE, then back: restoring a window activates it
                    time.sleep(0.15)
                    user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)
            for _ in range(8):
                if self.in_front(hwnd):
                    return True
                time.sleep(0.05)
        return self.in_front(hwnd)

    def _click_to_activate(self, hwnd: int) -> bool:
        """Click a spot of the window's top strip that belongs to the window and is not a control (the caption, the blank
        part of a tab strip). Real input is never refused the foreground."""
        r = self.window_rect(hwnd)
        w = r[2] - r[0]
        top = max(r[1], 0) + 10
        for frac in (0.5, 0.35, 0.65, 0.25, 0.75):
            x = int(r[0] + w * frac)
            pt = wt.POINT(x, top)
            under = user32.WindowFromPoint(pt)
            if not under or int(user32.GetAncestor(under, 2) or 0) != hwnd:   # GA_ROOT: something else is in front there
                continue
            try:
                el = self.iuia.ElementFromPoint(self.U.tagPOINT(x, top))
                if el.CurrentControlType not in (50032, 50033, 50037, 50026, 50020, 50021):
                    continue   # a button, a tab, a field: not a click to make blind
            except Exception:  # noqa: BLE001
                pass
            self.click_at(x, top)
            time.sleep(0.15)
            return True
        return False

    def close_window(self, hwnd: int) -> None:
        user32.PostMessageW(hwnd, 0x0010, 0, 0)   # WM_CLOSE

    def open(self, target: str) -> None:
        os.startfile(target)   # noqa: S606 - 'notepad', 'calc', a path, 'ms-settings:' (the user's request)

    # ------------------------------------------------------------------ UI Automation
    def snapshot(self, hwnd: int) -> list[dict]:
        """Every control-view element of the window in document order, one cross-process call."""
        U = self.U
        root = self.iuia.ElementFromHandle(hwnd)
        arr = root.FindAllBuildCache(4, self.tree, self.cache)   # TreeScope_Descendants
        out = []
        for i in range(arr.Length):
            el = arr.GetElement(i)
            try:
                r = el.CachedBoundingRectangle
                raw = {"el": el, "hwnd": hwnd, "ct": el.CachedControlType, "name": el.CachedName or "",
                       "rect": (r.left, r.top, r.right, r.bottom), "off": bool(el.CachedIsOffscreen),
                       "enabled": bool(el.CachedIsEnabled), "cls": el.CachedClassName or "",
                       "aid": el.CachedAutomationId or "", "help": el.CachedHelpText or "",
                       "pw": bool(el.GetCachedPropertyValue(P_PASSWORD) is True)}
            except Exception:  # noqa: BLE001 - an element that went away mid-query
                continue
            def prop(pid, kinds):
                try:
                    v = el.GetCachedPropertyValue(pid)
                except Exception:  # noqa: BLE001
                    return None
                return v if isinstance(v, kinds) else None   # (an unsupported property comes back as a COM marker)
            raw["value"] = prop(P_VALUE, str)
            raw["readonly"] = bool(prop(P_READONLY, bool))
            raw["toggle"] = prop(P_TOGGLE, int)
            raw["selected"] = prop(P_SELECTED, bool)
            raw["expand"] = prop(P_EXPAND, int)
            raw["avail"] = {k: bool(prop(p, bool)) for k, p in AVAIL.items()}
            out.append(raw)
        return out

    def _pattern(self, raw: dict, name: str, iface):
        p = raw["el"].GetCurrentPattern(PATTERN[name])
        return p.QueryInterface(iface)

    def invoke(self, raw: dict, select_first: bool = False) -> str:
        """Act on a control through its own UIA pattern (no mouse, no focus needed). Returns the pattern used."""
        U, a = self.U, raw["avail"]
        order = ["toggle", "select", "invoke", "expand"] if select_first else ["toggle", "invoke", "select", "expand"]
        for name in order:
            if not a.get(name):
                continue
            if name == "toggle":
                self._pattern(raw, "toggle", U.IUIAutomationTogglePattern).Toggle()
            elif name == "invoke":
                self._pattern(raw, "invoke", U.IUIAutomationInvokePattern).Invoke()
            elif name == "select":
                self._pattern(raw, "select", U.IUIAutomationSelectionItemPattern).Select()
            else:
                self._pattern(raw, "expand", U.IUIAutomationExpandCollapsePattern).Expand()
            return name
        raise RuntimeError("no UIA pattern to act with")

    def expand(self, raw: dict) -> bool:
        if raw["avail"].get("expand"):
            self._pattern(raw, "expand", self.U.IUIAutomationExpandCollapsePattern).Expand()
            return True
        return False

    def collapse(self, raw: dict) -> None:
        if raw["avail"].get("expand"):
            self._pattern(raw, "expand", self.U.IUIAutomationExpandCollapsePattern).Collapse()

    def get_value(self, raw: dict) -> str | None:
        """The control's text now (fresh, not from the snapshot)."""
        try:
            return self._pattern(raw, "value", self.U.IUIAutomationValuePattern).CurrentValue
        except Exception:  # noqa: BLE001
            pass
        try:
            tp = self._pattern(raw, "text", self.U.IUIAutomationTextPattern)
            return tp.DocumentRange.GetText(-1)
        except Exception:  # noqa: BLE001
            return None

    def set_value(self, raw: dict, text: str) -> None:
        self._pattern(raw, "value", self.U.IUIAutomationValuePattern).SetValue(text)

    def set_focus(self, raw: dict) -> None:
        raw["el"].SetFocus()

    def toggle_state(self, raw: dict) -> int | None:
        try:
            return self._pattern(raw, "toggle", self.U.IUIAutomationTogglePattern).CurrentToggleState
        except Exception:  # noqa: BLE001
            return None

    def scroll(self, raw: dict | None, hwnd: int, dy: int) -> bool:
        """Scroll down (dy > 0) or up: a scroll pattern when the control has one, else the mouse wheel over it."""
        if raw is not None and raw["avail"].get("scroll"):
            try:
                p = self._pattern(raw, "scroll", self.U.IUIAutomationScrollPattern)
                p.Scroll(0, 3 if dy > 0 else 0)   # ScrollAmount_NoAmount(0) horizontally, LargeIncrement(3)/LargeDecrement(0)
                return True
            except Exception:  # noqa: BLE001
                pass
        rect = raw["rect"] if raw else self.window_rect(hwnd)
        self.click_at((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2, move_only=True)
        self.wheel(-dy)
        return True

    def window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        r = wt.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    # ------------------------------------------------------------------ input
    @staticmethod
    def _send(*inputs: INPUT) -> None:
        arr = (INPUT * len(inputs))(*inputs)
        user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    @staticmethod
    def _key(vk: int = 0, scan: int = 0, up: bool = False, unicode: bool = False) -> INPUT:
        flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_UNICODE if unicode else 0) | \
                (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0)
        i = INPUT(type=1)
        i.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
        return i

    def keys(self, combo: str, times: int = 1, delay: float = 0.05) -> None:
        codes = parse_combo(combo)
        for _ in range(times):
            self._send(*[self._key(c) for c in codes], *[self._key(c, up=True) for c in reversed(codes)])
            time.sleep(delay)

    def type_text(self, text: str, delay: float = 0.004) -> None:
        """Type every character as a real key event (Unicode, so any layout and any language)."""
        for ch in text:
            if ch == "\r":
                continue
            if ch == "\n":
                self.keys("enter")
                continue
            data = ch.encode("utf-16-le")   # (a character outside the BMP is two 16-bit units)
            words = [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]
            self._send(*[self._key(0, w, unicode=True) for w in words],
                       *[self._key(0, w, up=True, unicode=True) for w in words])
            if delay:
                time.sleep(delay)

    def get_clipboard(self) -> str | None:
        for _ in range(10):
            if user32.OpenClipboard(None):
                break
            time.sleep(0.05)
        else:
            return None
        try:
            h = user32.GetClipboardData(13)   # CF_UNICODETEXT
            if not h:
                return None
            p = kernel32.GlobalLock(h)
            try:
                return ctypes.wstring_at(p)
            finally:
                kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()

    def set_clipboard(self, text: str | None) -> None:
        for _ in range(10):
            if user32.OpenClipboard(None):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the clipboard is in use by another program")
        try:
            user32.EmptyClipboard()
            if text is None:
                return
            data = (text + "\0").encode("utf-16-le")
            h = kernel32.GlobalAlloc(0x0002, len(data))   # GMEM_MOVEABLE
            p = kernel32.GlobalLock(h)
            ctypes.memmove(p, data, len(data))
            kernel32.GlobalUnlock(h)
            user32.SetClipboardData(13, h)
        finally:
            user32.CloseClipboard()

    def paste(self, text: str) -> None:
        """Ctrl+V with `text` on the clipboard; the clipboard's previous text is put back."""
        before = self.get_clipboard()
        self.set_clipboard(text)
        try:
            self.keys("ctrl+v")
            time.sleep(0.25)
        finally:
            self.set_clipboard(before)

    def click_at(self, x: int, y: int, button: str = "left", double: bool = False, move_only: bool = False) -> None:
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.03)
        if move_only:
            return
        down, up = MOUSEEVENTF[button]
        for _ in range(2 if double else 1):
            for flag in (down, up):
                i = INPUT(type=0)
                i.mi = MOUSEINPUT(0, 0, 0, flag, 0, 0)
                self._send(i)
                time.sleep(0.02)

    def wheel(self, clicks: int) -> None:
        i = INPUT(type=0)
        i.mi = MOUSEINPUT(0, 0, ctypes.c_ulong(120 * clicks).value, MOUSEEVENTF_WHEEL, 0, 0)
        self._send(i)

    def screenshot(self, path: str, rect: tuple[int, int, int, int] | None = None) -> str:
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=rect, all_screens=True) if rect else ImageGrab.grab(all_screens=True)
        img.save(path)
        return path
