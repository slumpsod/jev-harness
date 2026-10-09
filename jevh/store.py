"""The one way the harness opens its own SQLite stores (memory, usage, failures, reviews, notebook, history)."""
from __future__ import annotations

import os
import sqlite3

# Seconds a connection waits for another's write to end before failing with "database is locked". The stores are
# shared by every process on the workspace (the two dashboards, the CLI, a bench run) and SQLite's default of 5 s was
# short enough for a long read in one of them (the History tab listing every session) to fail a write in another.
BUSY_TIMEOUT = 30.0


def connect(path: str | os.PathLike) -> sqlite3.Connection:
    """A connection any thread may use: each store serializes its own use of it with a lock."""
    return sqlite3.connect(path, timeout=BUSY_TIMEOUT, check_same_thread=False)
