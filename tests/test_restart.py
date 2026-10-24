"""Restart now (Settings, Updates): /api/restart and the helper that starts the harness again (jevh/restart.py)."""
import asyncio
import subprocess
import sys
import time

import pytest

from jevh import restart
from jevh.web import server as web_server

from test_update import commit, repos, no_git_identity, no_install  # noqa: F401 - (fixtures)
from test_update import update
from test_web import BASE, auth, make_client


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def writer(path):
    """A command that proves it ran: it writes `path`."""
    return [sys.executable, "-c", f"open({str(path)!r}, 'w').write('ran')"]


# ---------------------------------------------------------------------- the endpoint
def test_restart_needs_a_way_to_stop(cfg):
    client, app = make_client(cfg)
    assert client.get("/api/version", headers=auth()).json()["can_restart"] is False
    r = client.post("/api/restart", headers=auth(Origin=BASE))
    assert r.status_code == 501 and not app.state.hub.busy.locked()


def test_restart_schedules_the_helper_then_stops(cfg, monkeypatch):
    client, app = make_client(cfg)
    stopped, scheduled = [], []
    app.state.stop = lambda: stopped.append(1)
    app.state.restart_cmd = lambda: ["python", "-m", "jevh", "app"]
    monkeypatch.setattr(web_server.restarter, "schedule", lambda port, log, cmd: scheduled.append((port, log, cmd)))
    assert client.get("/api/version", headers=auth()).json()["can_restart"] is True
    assert client.post("/api/restart", headers=auth(Origin="https://evil.example")).status_code == 403
    r = client.post("/api/restart", headers=auth(Origin=BASE))
    assert r.status_code == 200 and r.json()["log"].endswith("restart.log")
    assert scheduled == [(8740, cfg.workspace / "restart.log", ["python", "-m", "jevh", "app"])]
    assert wait_for(lambda: stopped, 5)
    assert app.state.hub.busy.locked()   # (kept: no turn starts while it goes down)


def test_no_restart_during_a_turn_or_a_scheduled_job(cfg, monkeypatch):
    client, app = make_client(cfg)
    hub = app.state.hub
    app.state.stop = lambda: None
    monkeypatch.setattr(web_server.restarter, "schedule", lambda *a, **k: pytest.fail("restarted"))
    hub.busy.acquire()
    try:
        r = client.post("/api/restart", headers=auth(Origin=BASE))
        assert r.status_code == 409 and "turn" in r.json()["detail"]
    finally:
        hub.busy.release()
    monkeypatch.setattr(hub.cron, "running", lambda: 1)
    r = client.post("/api/restart", headers=auth(Origin=BASE))
    assert r.status_code == 409 and "scheduled job" in r.json()["detail"]
    assert not hub.busy.locked()


def test_a_failed_schedule_lets_go_of_busy(cfg, monkeypatch):
    client, app = make_client(cfg)
    app.state.stop = lambda: pytest.fail("stopped without a helper to start it again")

    def boom(*a, **k):
        raise OSError("no python")
    monkeypatch.setattr(web_server.restarter, "schedule", boom)
    r = client.post("/api/restart", headers=auth(Origin=BASE))
    assert r.status_code == 500 and "no python" in r.json()["detail"] and not app.state.hub.busy.locked()


def test_end_streams_ends_every_open_event_stream(cfg):
    """(a restart: an open stream never ends by itself, and would hold the way down up for the whole grace period)"""
    _, app = make_client(cfg)
    hub = app.state.hub

    async def go():
        q = hub.subscribe()
        hub.end_streams()
        return q.behind, await asyncio.wait_for(q.get(), 1)
    behind, e = asyncio.run(go())
    assert behind and e["seq"] == -1


def test_code_changed_outside_the_dashboard_needs_a_restart(cfg, repos, monkeypatch):
    """`jevh update` in a terminal, or a git pull or merge, moves the code on disk too: the card offers Restart now."""
    user, _ = repos
    monkeypatch.setattr(update, "ROOT", user)
    client, _ = make_client(cfg)
    v = client.get("/api/version", headers=auth()).json()
    assert v["restart_needed"] is None and v["running"] == v["current"]["sha"]
    commit(user, "c.py", "c\n", "merged by hand")
    v = client.get("/api/version?refresh=1", headers=auth()).json()
    assert v["restart_needed"]["from"] == v["running"] and v["restart_needed"]["to"] == v["current"]["sha"] != v["running"]


# ---------------------------------------------------------------------- the helper
def sleeper(secs):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({secs})"])


def test_helper_waits_for_the_old_process_then_runs_the_command(tmp_path):
    old = sleeper(1.5)
    t0 = time.monotonic()
    p = restart.wait_and_run(old.pid, 0, writer(tmp_path / "ran.txt"), str(tmp_path))
    assert time.monotonic() - t0 >= 1.0 and old.poll() is not None   # (it waited)
    p.wait(20)
    assert (tmp_path / "ran.txt").read_text() == "ran"


def test_helper_ends_an_old_process_that_hangs(tmp_path):
    old = sleeper(60)
    try:
        p = restart.wait_and_run(old.pid, 0, writer(tmp_path / "ran.txt"), str(tmp_path), wait_exit=0.5)
        p.wait(20)
        assert old.wait(5) is not None and (tmp_path / "ran.txt").exists()
    finally:
        old.kill()


def test_schedule_runs_it_again_detached(tmp_path):
    """The real thing: a detached helper, which outlives the process that asked, and logs what it did."""
    old = sleeper(1)
    log = tmp_path / "logs" / "restart.log"
    cmd = writer(tmp_path / "ran.txt")
    assert restart.schedule(0, log, cmd, pid=old.pid) == cmd
    assert wait_for(lambda: (tmp_path / "ran.txt").exists(), 30), log.read_text(encoding="utf-8")
    text = log.read_text(encoding="utf-8")
    assert f"restart requested by pid {old.pid}" in text
    assert wait_for(lambda: "started pid" in log.read_text(encoding="utf-8"), 10)
    assert "restart-output.log" in log.read_text(encoding="utf-8") and (log.parent / "restart-output.log").exists()


def test_the_new_process_writes_to_its_own_file(tmp_path):
    out = tmp_path / "restart-output.log"
    p = restart.wait_and_run(0, 0, [sys.executable, "-c", "print('hello from the new one')"], str(tmp_path), out=out)
    p.wait(20)
    assert wait_for(lambda: "hello from the new one" in out.read_text(encoding="utf-8"), 5)


def test_command_is_this_process_again():
    cmd = restart.command()
    assert cmd[0] == sys.executable and cmd[1:] == sys.orig_argv[1:]


def test_the_app_comes_back_the_way_it_was(monkeypatch):
    from jevh import app as desktop
    monkeypatch.setattr(restart, "command", lambda: ["pythonw", "-m", "jevh", "app", "--hidden"])
    a = desktop.App({"role": "hub"})
    assert a.restart_cmd() == ["pythonw", "-m", "jevh", "app"]   # (no window yet: shown)

    class Win:
        pass
    a.window = Win()
    monkeypatch.setattr(a, "_main_visible", lambda: False)
    assert a.restart_cmd() == ["pythonw", "-m", "jevh", "app", "--hidden"]   # in the tray: back in the tray
    monkeypatch.setattr(a, "_main_visible", lambda: True)
    assert a.restart_cmd() == ["pythonw", "-m", "jevh", "app"]
