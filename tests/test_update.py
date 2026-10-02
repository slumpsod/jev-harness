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


def publish(dev, name, text, msg):
    commit(dev, name, text, msg)
    sh(dev, "push", "-q")


@pytest.fixture
def repos(tmp_path):
    """(the user's checkout, a second clone that pushes new versions to the shared 'GitHub' remote)"""
    origin, dev, user = tmp_path / "origin.git", tmp_path / "dev", tmp_path / "user"
    sh(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    sh(tmp_path, "clone", "-q", str(origin), str(dev))
    sh(dev, "checkout", "-q", "-b", "main")
    commit(dev, "a.py", "one\ntwo\nthree\nfour\nfive\n", "first")
    commit(dev, "b.py", "b1\n", "add b")
    sh(dev, "push", "-q", "-u", "origin", "main")
    sh(tmp_path, "clone", "-q", str(origin), str(user))
    return user, dev


@pytest.fixture(autouse=True)
def no_git_identity(tmp_path, monkeypatch):
    """Like a fresh machine: no global git config, so no user.name for the stash commits."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(autouse=True)
def no_install(monkeypatch):
    """No pip installs and no import check of the fake checkout unless a test asks; pip calls are recorded."""
    calls = []
    monkeypatch.setattr(update, "_pip", lambda root: calls.append(root))
    monkeypatch.setattr(update, "SMOKE", [])
    return calls


def head(repo):
    return sh(repo, "rev-parse", "HEAD")


def test_up_to_date(repos):
    user, _ = repos
    st = update.check(user)
    assert st["error"] is None and st["behind"] == 0 and not st["can_update"]
    assert st["branch"] == "main" and st["upstream"] == "origin/main" and st["current"]["subject"] == "add b"
    assert "up to date" in update.describe(st)
    assert update.apply(user)["updated"] is False


def test_sees_new_commits_updates_and_installs_new_dependencies(repos, no_install):
    user, dev = repos
    commit(dev, "b.py", "b2\n", "second")
    publish(dev, "pyproject.toml", "[project]\n", "new dependency")
    assert update.check(user, fetch=False)["behind"] == 0   # not fetched yet
    st = update.check(user)
    assert st["behind"] == 2 and st["can_update"] and st["deps_changed"]
    assert [c["subject"] for c in st["incoming"]] == ["new dependency", "second"]
    r = update.apply(user)
    assert r["updated"] and r["commits"] == 2 and r["files"] == 2 and r["kept"] == []
    assert (user / "b.py").read_text() == "b2\n" and no_install == [user]
    assert "New dependencies installed" in update.done(r) and "Restart" in update.done(r)
    assert update.check(user, fetch=False)["behind"] == 0


def test_edits_are_kept_across_an_update(repos):
    user, dev = repos
    publish(dev, "b.py", "b2\n", "second")
    (user / "a.py").write_text("my version\n", encoding="utf-8")   # a file the update doesn't touch
    st = update.check(user)
    assert st["dirty"] == ["a.py"] and st["can_update"] and "kept across the update" in update.describe(st)
    r = update.apply(user)
    assert r["kept"] == ["a.py"] and "Your edits to 1 file(s) were kept" in update.done(r)
    assert (user / "a.py").read_text() == "my version\n" and (user / "b.py").read_text() == "b2\n"
    assert sh(user, "stash", "list") == ""   # our stash entry is gone; a copy stays at the ref
    assert sh(user, "show", f"{update.EDITS_REF}:a.py") == "my version"


def test_edits_to_a_file_the_update_also_changes_are_merged(repos):
    user, dev = repos
    publish(dev, "a.py", "one\ntwo\nthree\nfour\nFIVE (upstream)\n", "upstream edit")
    (user / "a.py").write_text("ONE (mine)\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    assert update.check(user)["can_update"]
    update.apply(user)
    assert (user / "a.py").read_text() == "ONE (mine)\ntwo\nthree\nfour\nFIVE (upstream)\n"


def test_clashing_edits_stop_the_update_and_nothing_changes(repos):
    user, dev = repos
    publish(dev, "a.py", "one\nTWO (upstream)\nthree\nfour\nfive\n", "upstream edit")
    (user / "a.py").write_text("one\nTWO (mine)\nthree\nfour\nfive\n", encoding="utf-8")
    before = head(user)
    st = update.check(user)
    assert st["conflicts"] == ["a.py"] and not st["can_update"] and "clash" in update.describe(st)
    with pytest.raises(update.UpdateError, match="clash"):
        update.apply(user)
    assert head(user) == before and (user / "a.py").read_text() == "one\nTWO (mine)\nthree\nfour\nfive\n"
    assert sh(user, "stash", "list") == ""


def test_a_new_file_the_update_would_overwrite_stops_it(repos):
    user, dev = repos
    publish(dev, "skill.md", "theirs\n", "add skill")
    (user / "skill.md").write_text("mine\n", encoding="utf-8")
    st = update.check(user)
    assert st["in_way"] == ["skill.md"] and not st["can_update"]
    with pytest.raises(update.UpdateError, match="already have"):
        update.apply(user)
    assert (user / "skill.md").read_text() == "mine\n"


def test_a_version_that_doesnt_start_is_rolled_back_with_the_edits(repos, monkeypatch):
    user, dev = repos
    publish(dev, "app.py", "x = 1\n", "app")
    update.apply(user)
    publish(dev, "app.py", "import no_such_module_anywhere\n", "broken release")
    (user / "b.py").write_text("my b\n", encoding="utf-8")
    monkeypatch.setattr(update, "SMOKE", ["app"])
    before = head(user)
    with pytest.raises(update.UpdateError, match="doesn't start(.|\n)*Rolled back"):
        update.apply(user)
    assert head(user) == before and (user / "app.py").read_text() == "x = 1\n"
    assert (user / "b.py").read_text() == "my b\n" and sh(user, "stash", "list") == ""


def test_failed_dependency_install_is_rolled_back(repos, monkeypatch):
    user, dev = repos
    publish(dev, "pyproject.toml", "[project]\n", "new dependency")
    calls = []

    def pip(root):
        calls.append(root)
        if len(calls) == 1:
            raise update.UpdateError("installing the new dependencies failed: no such package")

    monkeypatch.setattr(update, "_pip", pip)
    before = head(user)
    with pytest.raises(update.UpdateError, match="Rolled back"):
        update.apply(user)
    assert head(user) == before and not (user / "pyproject.toml").exists()
    assert len(calls) == 2   # the old version's dependencies put back


def test_tests_must_pass_with_test(repos):
    user, dev = repos
    publish(dev, "test_x.py", "def test_ok():\n    assert True\n", "a passing test")
    assert update.apply(user, test=True)["updated"]
    publish(dev, "test_x.py", "def test_ok():\n    assert False\n", "a failing test")
    before = head(user)
    with pytest.raises(update.UpdateError, match="tests fail(.|\n)*Rolled back"):
        update.apply(user, test=True)
    assert head(user) == before


def test_undo_goes_back_keeping_edits(repos):
    user, dev = repos
    before = head(user)
    publish(dev, "b.py", "b2\n", "second")
    update.apply(user)
    st = update.check(user, fetch=False)
    assert st["previous"]["sha"] == sh(user, "rev-parse", "--short", before) and "--undo" in update.describe(st)
    (user / "a.py").write_text("edited after\n", encoding="utf-8")
    r = update.undo(user)
    assert r["undone"] and head(user) == before and (user / "b.py").read_text() == "b1\n"
    assert (user / "a.py").read_text() == "edited after\n" and "back on" in update.done(r)
    st = update.check(user, fetch=False)
    assert st["previous"] is None and st["behind"] == 1   # the update is offered again
    with pytest.raises(update.UpdateError, match="nothing to undo"):
        update.undo(user)


def test_diverged_checkout_is_not_merged(repos):
    user, dev = repos
    publish(dev, "b.py", "b2\n", "second")
    commit(user, "c.py", "mine\n", "local work")
    st = update.check(user)
    assert st["ahead"] == 1 and st["behind"] == 1 and not st["can_update"] and "aren't on origin/main" in st["error"]
    with pytest.raises(update.UpdateError):
        update.apply(user)


def test_not_a_checkout_or_no_upstream(tmp_path, repos):
    assert "not a git checkout" in update.check(tmp_path / "nowhere")["error"]
    user, _ = repos
    sh(user, "checkout", "-q", "-b", "feature")
    assert "doesn't track" in update.check(user)["error"]


def test_cli_update_and_undo(repos, monkeypatch, capsys):
    user, dev = repos
    publish(dev, "b.py", "b2\n", "second")
    monkeypatch.setattr(update, "ROOT", user)
    assert cli_main(["--update"]) == 0
    out = capsys.readouterr().out
    assert "1 new commit" in out and "updated" in out and (user / "b.py").read_text() == "b2\n"
    assert cli_main(["update", "--undo"]) == 0
    assert "back on" in capsys.readouterr().out and (user / "b.py").read_text() == "b1\n"


def test_dashboard_version_update_and_undo(cfg, repos, monkeypatch):
    user, dev = repos
    monkeypatch.setattr(update, "ROOT", user)
    client, _ = make_client(cfg)
    v = client.get("/api/version", headers=auth()).json()
    assert v["behind"] == 0 and v["restart_needed"] is None and v["previous"] is None
    publish(dev, "b.py", "b2\n", "second")
    assert client.get("/api/version", headers=auth()).json()["behind"] == 0   # cached
    v = client.get("/api/version?refresh=1", headers=auth()).json()
    assert v["behind"] == 1 and v["incoming"][0]["subject"] == "second" and v["can_update"]
    assert client.post("/api/update", headers=auth(Origin="https://evil.example")).status_code == 403
    r = client.post("/api/update", json={"test": False}, headers=auth(Origin=BASE)).json()
    assert r["updated"] and r["restart_needed"]["to"] == r["to"] and "Restart" in r["message"]
    v = client.get("/api/version", headers=auth()).json()
    assert v["behind"] == 0 and v["restart_needed"]["to"] == r["to"] and v["previous"]
    r = client.post("/api/update/undo", headers=auth(Origin=BASE)).json()
    assert r["undone"] and r["restart_needed"] is None   # back on the code the server runs
    assert client.post("/api/update/undo", headers=auth(Origin=BASE)).status_code == 409


def test_jevh_subcommands_dispatch(monkeypatch):
    import jevh.web.__main__ as web
    seen = []
    monkeypatch.setattr(web, "main", lambda argv: seen.append(("dashboard", argv)))
    monkeypatch.setattr(update, "main", lambda argv: seen.append(("update", argv)) or 0)
    cli_main(["dashboard", "--no-browser", "--port", "8750"])
    assert cli_main(["update", "--check"]) == 0
    assert seen == [("dashboard", ["--no-browser", "--port", "8750"]), ("update", ["--check"])]


def test_stable_channel_is_followed_once_it_exists(repos):
    user, dev = repos
    sh(dev, "push", "-q", "origin", "main:stable")          # CI marks the current main as stable
    publish(dev, "b.py", "b2\n", "untested on main")
    st = update.check(user)
    assert st["channel"] == "stable" and st["upstream"] == "origin/stable" and st["behind"] == 0
    assert set(st["channels"]) >= {"main", "stable"}
    st = update.check(user, channel="main")               # this once
    assert st["upstream"] == "origin/main" and st["behind"] == 1
    sh(dev, "push", "-q", "origin", "main:stable")          # the tests passed: stable moves
    r = update.apply(user)
    assert r["updated"] and (user / "b.py").read_text() == "b2\n"


def test_set_channel_and_unknown_channels(repos, monkeypatch, capsys):
    user, dev = repos
    sh(dev, "push", "-q", "origin", "main:stable")
    monkeypatch.setattr(update, "ROOT", user)
    assert cli_main(["update", "--set-channel", "main"]) == 0
    assert update.get_channel(user) == "main" and "following main" in capsys.readouterr().out
    assert cli_main(["update", "--set-channel", "nightly"]) == 1
    assert update.get_channel(user) == "main" and "no 'nightly' channel" in capsys.readouterr().out
    assert cli_main(["update", "--set-channel", "default"]) == 0
    assert update.get_channel(user) is None and update.check(user)["channel"] == "stable"


def harness_repo(repos):
    """The user's checkout as a harness install: a config.yaml whose data lives in ws/ (ignored by git)."""
    user, dev = repos
    commit(dev, ".gitignore", "ws/\nconfig.local.yaml\n", "ignore data")
    publish(dev, "config.yaml", "workspace: ws\nskills_dirs: [skills]\n", "config")
    update.apply(user)
    import sqlite3
    (user / "ws" / "memory").mkdir(parents=True)
    with sqlite3.connect(user / "ws" / "memory" / "memory.db") as db:
        db.execute("create table facts (t text)")
        db.execute("insert into facts values ('likes tea')")
    db.close()
    (user / "ws" / "sessions" / "s1").mkdir(parents=True)
    (user / "ws" / "sessions" / "s1" / "big.txt").write_text("session files", encoding="utf-8")
    (user / "config.local.yaml").write_text("models: {}\n", encoding="utf-8")
    return user, dev


def test_data_is_snapshotted_before_an_update_and_can_be_restored(repos, monkeypatch, capsys):
    import sqlite3
    from jevh import snapshot
    user, dev = harness_repo(repos)
    publish(dev, "b.py", "b2\n", "second")
    r = update.apply(user)
    assert r["snapshot"] and "Data snapshot" in update.done(r)
    snap = user / "ws" / "snapshots" / r["snapshot"]
    assert (snap / "workspace" / "memory" / "memory.db").is_file() and (snap / "config.local.yaml").is_file()
    assert not (snap / "workspace" / "sessions").exists()   # session folders aren't copied
    with sqlite3.connect(user / "ws" / "memory" / "memory.db") as db:   # the new version changes the data
        db.execute("delete from facts")
    db.close()
    monkeypatch.setattr(update, "ROOT", user)
    assert cli_main(["update", "--snapshots"]) == 0 and r["snapshot"] in capsys.readouterr().out
    assert cli_main(["update", "--restore-data", r["snapshot"]]) == 0
    with sqlite3.connect(user / "ws" / "memory" / "memory.db") as db:
        assert db.execute("select t from facts").fetchall() == [("likes tea",)]
    db.close()
    assert (user / "ws" / "sessions" / "s1" / "big.txt").is_file()   # left alone
    names = [m["reason"] for m in snapshot.listing(update._config(user))]
    assert names[0] == "before-restore"   # the restore itself can be undone


def test_undo_keeps_data_and_says_how_to_restore_it(repos):
    user, dev = harness_repo(repos)
    publish(dev, "b.py", "b2\n", "second")
    r = update.apply(user)
    u = update.undo(user)
    assert u["pre_update_snapshot"] == r["snapshot"] and f"--restore-data {r['snapshot']}" in update.done(u)


def test_snapshots_are_pruned(repos):
    from jevh import snapshot
    user, _ = harness_repo(repos)
    cfg = update._config(user)
    for i in range(snapshot.KEEP + 2):
        snapshot.take(cfg, f"t{i}")
    assert len(snapshot.listing(cfg)) == snapshot.KEEP
