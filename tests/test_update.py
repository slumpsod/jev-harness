"""Updating the harness from GitHub (jevh/update.py), against a local bare repo standing in for GitHub."""
import subprocess

import pytest

from jevh import update
from jevh.cli import main as cli_main

from test_web import BASE, auth, make_client


def sh(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, name, text, msg):
    (repo / name).write_text(text, encoding="utf-8")
    sh(repo, "add", name)
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)


@pytest.fixture
def repos(tmp_path):
    """(the user's checkout, a second clone that pushes new versions to the shared 'GitHub' remote)"""
    origin, dev, user = tmp_path / "origin.git", tmp_path / "dev", tmp_path / "user"
    sh(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    sh(tmp_path, "clone", "-q", str(origin), str(dev))
    sh(dev, "checkout", "-q", "-b", "main")
    commit(dev, "a.py", "v1\n", "first")
    sh(dev, "push", "-q", "-u", "origin", "main")
    sh(tmp_path, "clone", "-q", str(origin), str(user))
    return user, dev


def test_up_to_date(repos):
    user, _ = repos
    st = update.check(user)
    assert st["error"] is None and st["behind"] == 0 and not st["can_update"]
    assert st["branch"] == "main" and st["upstream"] == "origin/main" and st["current"]["subject"] == "first"
    assert "up to date" in update.describe(st)
    assert update.apply(user)["updated"] is False


def test_sees_new_commits_and_fast_forwards(repos):
    user, dev = repos
    commit(dev, "a.py", "v2\n", "second")
    commit(dev, "pyproject.toml", "[project]\n", "new dependency")
    sh(dev, "push", "-q")
    assert update.check(user, fetch=False)["behind"] == 0   # not fetched yet
    st = update.check(user)
    assert st["behind"] == 2 and st["can_update"] and st["deps_changed"]
    assert [c["subject"] for c in st["incoming"]] == ["new dependency", "second"]
    r = update.apply(user)
    assert r["updated"] and r["commits"] == 2 and r["files"] == 2
    assert (user / "a.py").read_text() == "v2\n"
    assert "Restart" in update.done(r) and "pip install" in update.done(r)
    assert update.check(user, fetch=False)["behind"] == 0


def test_local_edits_in_the_way_stop_the_update(repos):
    user, dev = repos
    commit(dev, "a.py", "v2\n", "second")
    sh(dev, "push", "-q")
    (user / "a.py").write_text("my edit\n", encoding="utf-8")
    assert update.check(user)["dirty"] == ["a.py"]
    with pytest.raises(update.UpdateError, match="stash"):
        update.apply(user)
    assert (user / "a.py").read_text() == "my edit\n"   # nothing lost


def test_diverged_checkout_is_not_merged(repos):
    user, dev = repos
    commit(dev, "a.py", "v2\n", "second")
    sh(dev, "push", "-q")
    commit(user, "b.py", "mine\n", "local work")
    st = update.check(user)
    assert st["ahead"] == 1 and st["behind"] == 1 and not st["can_update"] and "local commit" in st["error"]
    with pytest.raises(update.UpdateError):
        update.apply(user)


def test_not_a_checkout_or_no_upstream(tmp_path, repos):
    assert "not a git checkout" in update.check(tmp_path / "nowhere")["error"]
    user, _ = repos
    sh(user, "checkout", "-q", "-b", "feature")
    assert "doesn't track" in update.check(user)["error"]


def test_cli_update_flag(repos, monkeypatch, capsys):
    user, dev = repos
    commit(dev, "a.py", "v2\n", "second")
    sh(dev, "push", "-q")
    monkeypatch.setattr(update, "ROOT", user)
    assert cli_main(["--update"]) == 0
    out = capsys.readouterr().out
    assert "1 new commit" in out and "updated" in out and (user / "a.py").read_text() == "v2\n"


def test_dashboard_version_and_update(cfg, repos, monkeypatch):
    user, dev = repos
    monkeypatch.setattr(update, "ROOT", user)
    client, _ = make_client(cfg)
    v = client.get("/api/version", headers=auth()).json()
    assert v["behind"] == 0 and v["restart_needed"] is None
    commit(dev, "a.py", "v2\n", "second")
    sh(dev, "push", "-q")
    assert client.get("/api/version", headers=auth()).json()["behind"] == 0   # cached
    v = client.get("/api/version?refresh=1", headers=auth()).json()
    assert v["behind"] == 1 and v["incoming"][0]["subject"] == "second"
    assert client.post("/api/update", headers=auth(Origin="https://evil.example")).status_code == 403
    r = client.post("/api/update", headers=auth(Origin=BASE)).json()
    assert r["updated"] and r["restart_needed"]["commits"] == 1 and "Restart" in r["message"]
    v = client.get("/api/version", headers=auth()).json()
    assert v["behind"] == 0 and v["restart_needed"]["to"] == r["to"]


def test_jevh_subcommands_dispatch(monkeypatch):
    import jevh.web.__main__ as web
    seen = []
    monkeypatch.setattr(web, "main", lambda argv: seen.append(("dashboard", argv)))
    monkeypatch.setattr(update, "main", lambda argv: seen.append(("update", argv)) or 0)
    cli_main(["dashboard", "--no-browser", "--port", "8750"])
    assert cli_main(["update", "--check"]) == 0
    assert seen == [("dashboard", ["--no-browser", "--port", "8750"]), ("update", ["--check"])]
