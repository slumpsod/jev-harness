"""Sign-in for the dashboard and the devices paired to it (workspace/accounts.db).

- Accounts: a name and a password (scrypt). Every account on a hub is an owner of it: there is one assistant, one memory,
  one set of devices. Accounts are made on the hub machine itself (`jevh users add`, or the dashboard opened on this machine).
- Sign-in sessions: a random token in an HttpOnly cookie (`jevh_session`); only its SHA-256 is stored. They outlive a
  dashboard restart (the old per-launch `?key=` link did not), last SESSION_DAYS, and can be ended one by one.
- Invites: a one-time link (INVITE_MINUTES long) to make an account from any device. Made on the hub computer
  (`jevh users invite`, or the dashboard there) or by someone signed in; whoever opens it picks their own name and
  password, so nobody has to be at the hub, and no password is ever passed along. Only the link's hash is stored.
- Devices: a computer the assistant may use (jevh/devices). A signed-in user makes a pairing code; the device trades the
  code once, within PAIR_MINUTES, for its own key, which it sends on every connection. Only the key's hash is stored, so a
  copy of accounts.db lets nobody in. Revoking a device ends its key (and the hub drops its connection).

Nothing here is a secret by itself, but the file says who may run code on which computer: it is created readable by
this user only (as workspace/settings.yaml is).
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SESSION_DAYS = 30
PAIR_MINUTES = 10
INVITE_MINUTES = 30
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O or 1/I: read aloud or typed from another screen
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}$")
MIN_PASSWORD = 8
# scrypt at n=2**14, r=8: ~16 MB and ~50 ms a check. A login is rare; a guess has to pay it too.
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "maxmem": 64 * 1024 * 1024, "dklen": 32}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY COLLATE NOCASE, pw TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user TEXT NOT NULL, created REAL NOT NULL,
    expires REAL NOT NULL, seen REAL NOT NULL, agent TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS devices (id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE, owner TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE, platform TEXT NOT NULL DEFAULT '', created REAL NOT NULL, seen REAL NOT NULL DEFAULT 0,
    revoked REAL);
CREATE TABLE IF NOT EXISTS pair_codes (code TEXT PRIMARY KEY, user TEXT NOT NULL, expires REAL NOT NULL, used REAL);
CREATE TABLE IF NOT EXISTS invites (code TEXT PRIMARY KEY, by TEXT NOT NULL, expires REAL NOT NULL, used REAL);
"""


class AccountError(ValueError):
    """Something the user asked for that cannot be done, in words they can act on."""


@dataclass
class Device:
    id: str
    name: str
    owner: str
    platform: str
    created: float
    seen: float
    revoked: float | None

    def json(self) -> dict:
        return {"id": self.id, "name": self.name, "owner": self.owner, "platform": self.platform,
                "created": self.created, "seen": self.seen, "revoked": self.revoked}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, want = stored.split("$")
        if kind != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                            maxmem=SCRYPT["maxmem"], dklen=len(want) // 2)
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(dk.hex(), want)


def clean_name(name: str, what: str = "name") -> str:
    name = " ".join(str(name or "").split())
    if not NAME_RE.match(name):
        raise AccountError(f"A {what} is 1 to 40 letters, digits, spaces, dots, dashes or underscores, starting with a "
                           f"letter or digit.")
    return name


class Accounts:
    """The accounts, sign-in sessions, devices and pairing codes of one hub. Thread-safe (one connection, one lock)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        if new:
            try:
                os.chmod(self.path, 0o600)   # (Windows keeps only the read-only bit: the folder's ACL is what guards it there)
            except OSError:
                pass
        self._failures: dict[str, list[float]] = {}   # who -> recent failed attempts (sign-in, pairing)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ users
    def has_users(self) -> bool:
        return bool(self._q("SELECT 1 FROM users LIMIT 1"))

    def users(self) -> list[dict]:
        return [{"name": n, "created": c} for n, c in self._q("SELECT name, created FROM users ORDER BY created")]

    def add_user(self, name: str, password: str) -> str:
        name = clean_name(name, "user name")
        if len(password or "") < MIN_PASSWORD:
            raise AccountError(f"Use a password of at least {MIN_PASSWORD} characters.")
        if self._q("SELECT 1 FROM users WHERE name = ?", (name,)):
            raise AccountError(f"There is already an account called {name}.")
        self._q("INSERT INTO users (name, pw, created) VALUES (?, ?, ?)", (name, hash_password(password), time.time()))
        return name

    def set_password(self, name: str, password: str) -> None:
        if len(password or "") < MIN_PASSWORD:
            raise AccountError(f"Use a password of at least {MIN_PASSWORD} characters.")
        if not self._q("SELECT 1 FROM users WHERE name = ?", (name,)):
            raise AccountError(f"No account called {name}.")
        self._q("UPDATE users SET pw = ? WHERE name = ?", (hash_password(password), name))
        self._q("DELETE FROM sessions WHERE user = ?", (name,))   # a new password signs every browser out

    def remove_user(self, name: str) -> None:
        if not self._q("SELECT 1 FROM users WHERE name = ?", (name,)):
            raise AccountError(f"No account called {name}.")
        self._q("DELETE FROM users WHERE name = ?", (name,))
        self._q("DELETE FROM sessions WHERE user = ?", (name,))

    def verify(self, name: str, password: str, who: str = "") -> str | None:
        """The account's own spelling of its name when the password is right; None otherwise (also while `who` is
        throttled). A wrong name costs the same scrypt as a wrong password, so timing does not tell them apart."""
        if self.throttled(f"login:{who}"):
            return None
        rows = self._q("SELECT name, pw FROM users WHERE name = ?", (str(name or "").strip(),))
        stored = rows[0][1] if rows else hash_password("x" * MIN_PASSWORD, b"\0" * 16)
        ok = check_password(password or "", stored) and bool(rows)
        if not ok:
            self.failed(f"login:{who}")
            return None
        return rows[0][0]

    # ------------------------------------------------------------------ throttling
    def throttled(self, who: str, limit: int = 8, window: float = 300.0) -> bool:
        """More than `limit` failures from `who` in the last `window` seconds."""
        now = time.time()
        with self._lock:
            recent = [t for t in self._failures.get(who, []) if now - t < window]
            self._failures[who] = recent
            return len(recent) >= limit

    def failed(self, who: str) -> None:
        with self._lock:
            self._failures.setdefault(who, []).append(time.time())

    # ------------------------------------------------------------------ sign-in sessions
    def start_session(self, user: str, agent: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._q("INSERT INTO sessions (token, user, created, expires, seen, agent) VALUES (?, ?, ?, ?, ?, ?)",
                (_sha(token), user, now, now + SESSION_DAYS * 86400, now, agent[:200]))
        return token

    def session_user(self, token: str | None) -> str | None:
        if not token:
            return None
        rows = self._q("SELECT s.user, s.expires, s.seen FROM sessions s JOIN users u ON u.name = s.user "
                       "WHERE s.token = ?", (_sha(token),))
        if not rows:
            return None
        user, expires, seen = rows[0]
        now = time.time()
        if expires < now:
            self._q("DELETE FROM sessions WHERE token = ?", (_sha(token),))
            return None
        if now - seen > 3600:   # (written at most hourly: every request would make the page a writer)
            self._q("UPDATE sessions SET seen = ? WHERE token = ?", (now, _sha(token)))
        return user

    def end_session(self, token: str | None) -> None:
        if token:
            self._q("DELETE FROM sessions WHERE token = ?", (_sha(token),))

    # ------------------------------------------------------------------ invites
    def invite(self, by: str, minutes: float = INVITE_MINUTES) -> tuple[str, float]:
        """A one-time token for an invite link (/setup?invite=...): whoever has it may make one account."""
        token = secrets.token_urlsafe(18)
        expires = time.time() + minutes * 60
        self._q("DELETE FROM invites WHERE expires < ?", (time.time() - 86400,))
        self._q("INSERT INTO invites (code, by, expires) VALUES (?, ?, ?)", (_sha(token), by, expires))
        return token, expires

    def invite_ok(self, token: str | None) -> bool:
        rows = self._q("SELECT expires, used FROM invites WHERE code = ?", (_sha(str(token or "")),))
        return bool(rows) and rows[0][1] is None and rows[0][0] >= time.time()

    def accept_invite(self, token: str, name: str, password: str, who: str = "") -> str:
        """Make the invited account (the invite is then used up) -> its name."""
        if self.throttled(f"invite:{who}"):
            raise AccountError("Too many tries: wait a few minutes.")
        if not self.invite_ok(token):
            self.failed(f"invite:{who}")
            raise AccountError("That invite link is wrong, used or expired: ask for a new one.")
        name = self.add_user(name, password)   # (checks the name and the password first: a bad one keeps the invite)
        self._q("UPDATE invites SET used = ? WHERE code = ?", (time.time(), _sha(str(token))))
        return name

    # ------------------------------------------------------------------ devices
    def pair_code(self, user: str) -> tuple[str, float]:
        """A one-time code (8 characters, PAIR_MINUTES long) a device trades for its key."""
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        expires = time.time() + PAIR_MINUTES * 60
        self._q("DELETE FROM pair_codes WHERE expires < ?", (time.time() - 86400,))
        self._q("INSERT INTO pair_codes (code, user, expires) VALUES (?, ?, ?)", (_sha(code), user, expires))
        return f"{code[:4]}-{code[4:]}", expires

    def redeem(self, code: str, name: str, platform: str = "", who: str = "") -> tuple[Device, str]:
        """Trade a pairing code for a device and its key (shown once: only its hash is kept). A name already paired
        and not revoked gets a number, so two computers called the same never share a key."""
        if self.throttled(f"pair:{who}"):
            raise AccountError("Too many wrong codes: wait a few minutes.")
        flat = re.sub(r"[^A-Z0-9]", "", str(code or "").upper())
        rows = self._q("SELECT user, expires, used FROM pair_codes WHERE code = ?", (_sha(flat),))
        if not rows or rows[0][2] is not None or rows[0][1] < time.time():
            self.failed(f"pair:{who}")
            raise AccountError("That pairing code is wrong, used or expired: make a new one in the dashboard.")
        user = rows[0][0]
        self._q("UPDATE pair_codes SET used = ? WHERE code = ?", (time.time(), _sha(flat)))
        base = clean_name(name or "device", "device name")
        taken = {n.lower() for (n,) in self._q("SELECT name FROM devices WHERE revoked IS NULL")}
        final, k = base, 2
        while final.lower() in taken:
            final, k = f"{base}-{k}", k + 1
        key = "jevd_" + secrets.token_urlsafe(32)
        dev = Device(id=secrets.token_hex(8), name=final, owner=user, platform=str(platform)[:60],
                     created=time.time(), seen=0.0, revoked=None)
        self._q("INSERT INTO devices (id, name, owner, key, platform, created, seen) VALUES (?, ?, ?, ?, ?, ?, 0)",
                (dev.id, dev.name, dev.owner, _sha(key), dev.platform, dev.created))
        return dev, key

    def device_for_key(self, key: str | None) -> Device | None:
        if not key:
            return None
        rows = self._q("SELECT id, name, owner, platform, created, seen, revoked FROM devices WHERE key = ?", (_sha(key),))
        if not rows or rows[0][6] is not None:
            return None
        return Device(*rows[0])

    def devices(self, revoked: bool = False) -> list[Device]:
        sql = "SELECT id, name, owner, platform, created, seen, revoked FROM devices"
        if not revoked:
            sql += " WHERE revoked IS NULL"
        return [Device(*r) for r in self._q(sql + " ORDER BY created")]

    def device(self, ref: str) -> Device | None:
        """By id, or by name (live devices only)."""
        rows = self._q("SELECT id, name, owner, platform, created, seen, revoked FROM devices WHERE id = ? OR "
                       "(name = ? AND revoked IS NULL) ORDER BY revoked IS NULL DESC LIMIT 1", (ref, ref))
        return Device(*rows[0]) if rows else None

    def seen(self, device_id: str) -> None:
        self._q("UPDATE devices SET seen = ? WHERE id = ?", (time.time(), device_id))

    def rename(self, device_id: str, name: str) -> str:
        name = clean_name(name, "device name")
        if self._q("SELECT 1 FROM devices WHERE name = ? AND id != ? AND revoked IS NULL", (name, device_id)):
            raise AccountError(f"Another device is already called {name}.")
        self._q("UPDATE devices SET name = ? WHERE id = ?", (name, device_id))
        return name

    def revoke(self, device_id: str) -> None:
        self._q("UPDATE devices SET revoked = ? WHERE id = ? AND revoked IS NULL", (time.time(), device_id))


# ---------------------------------------------------------------------- jevh users ...

def main(argv: list[str] | None = None) -> None:
    import argparse
    import getpass
    import sys

    from .config import load_config
    ap = argparse.ArgumentParser(prog="jevh users", description="Accounts that sign in to this hub's dashboard from other "
                                 "devices. Run on the hub computer; the dashboard need not be stopped.")
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the accounts, and the devices paired to this hub")
    p = sub.add_parser("add", help="make an account (asks for its password)")
    p.add_argument("name")
    p = sub.add_parser("password", help="set an account's password (signs it out everywhere)")
    p.add_argument("name")
    p = sub.add_parser("remove", help="delete an account (its devices stay paired)")
    p.add_argument("name")
    p = sub.add_parser("invite", help="a one-time link to make an account from any device (no need to be at the hub)")
    p.add_argument("--hub", help="the hub's address as other devices reach it, e.g. https://desktop.tailnet.ts.net:10000")
    p.add_argument("--minutes", type=float, default=INVITE_MINUTES)
    a = ap.parse_args(argv)
    acc = Accounts(load_config(a.config).workspace / "accounts.db")

    def ask() -> str:
        pw = getpass.getpass("Password: ")
        if pw != getpass.getpass("Again: "):
            sys.exit("The two don't match.")
        return pw
    try:
        if a.cmd == "list":
            users = acc.users()
            print("accounts:", ", ".join(u["name"] for u in users) or "none (other devices use the per-launch key link)")
            for d in acc.devices():
                seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.seen)) if d.seen else "never"
                print(f"device {d.name}  ({d.platform or '?'}; paired by {d.owner}; last seen {seen})")
        elif a.cmd == "add":
            print(f"Made the account {acc.add_user(a.name, ask())}. From now on other devices sign in to the dashboard "
                  f"(the ?key= link stops working).")
        elif a.cmd == "password":
            acc.set_password(a.name, ask())
            print("Password changed; every browser signed in to it was signed out.")
        elif a.cmd == "invite":
            token, expires = acc.invite("hub computer", a.minutes)
            hub = (a.hub or "<the hub's address, e.g. https://this-pc.tailnet.ts.net:10000>").rstrip("/")
            print(f"{hub}/setup?invite={token}\n(works once, until {time.strftime('%H:%M', time.localtime(expires))}; "
                  f"whoever opens it picks a name and password. Keep it private.)")
        elif a.cmd == "remove":
            acc.remove_user(a.name)
            print(f"Removed {a.name}." + ("" if acc.has_users() else " No accounts are left: the ?key= link works again."))
    except AccountError as e:
        sys.exit(str(e))
