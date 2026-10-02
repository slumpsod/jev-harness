"""A persona or skill that can't be used is skipped with a note instead of stopping the harness;
your own folder (JEVH_HOME) is laid over the checkout's config, personas and skills."""
from jevh.config import ROOT, load_config
from jevh.jev import Jev
from jevh.orchestrator import Harness
from jevh.personas import load_personas
from jevh.skills import load_roster

from conftest import FakeSystemOne

GOOD = "description: helps\nsystem_prompt: hi\nmodel: {provider: lmstudio, model: m}\n"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_broken_persona_files_are_skipped_with_a_reason(tmp_path):
    d = tmp_path / "personas"
    write(d / "good.yaml", GOOD)
    write(d / "typo.yaml", GOOD + "colour: blue\n")
    write(d / "broken.yaml", "description: [unclosed\n")
    write(d / "nodesc.yaml", "system_prompt: hi\nmodel: {provider: lmstudio, model: m}\n")
    write(d / "notalist.yaml", GOOD + "tools: web\n")
    problems = []
    personas = load_personas(d, problems)
    assert sorted(personas) == ["good", "typo"]
    by = {p["name"]: p for p in problems}
    assert not by["typo"]["skipped"] and "colour" in by["typo"]["error"]
    assert by["broken"]["skipped"] and "nodesc" in by and "must be a list" in by["notalist"]["error"]


def test_the_first_folder_wins_so_yours_replace_built_ins(tmp_path):
    mine = write(tmp_path / "mine" / "daily_assistant.yaml", GOOD.replace("helps", "my version")).parent
    personas = load_personas([mine, ROOT / "personas"])
    assert personas["daily_assistant"].description == "my version" and "fitness_coach" in personas


def test_broken_skills_are_skipped_with_a_reason(tmp_path):
    write(tmp_path / "ok" / "SKILL.md", "---\nname: ok\ndescription: fine\n---\nbody")
    write(tmp_path / "bad" / "SKILL.md", "---\nname: bad\ndescription: [oops\n---\nbody")
    write(tmp_path / "empty" / "SKILL.md", "---\nname: empty\n---\nbody")
    problems = []
    assert list(load_roster([tmp_path], problems=problems)) == ["ok"]
    assert {p["name"] for p in problems} == {"bad", "empty"} and all(p["skipped"] for p in problems)


def test_harness_starts_without_a_persona_on_an_unknown_provider(cfg, tmp_path):
    write(tmp_path / "p" / "ghost.yaml", GOOD.replace("lmstudio", "nowhere"))
    write(tmp_path / "p" / "daily_assistant.yaml", "{not yaml")
    cfg.personas_dirs = [tmp_path / "p", ROOT / "personas"]
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()))
    assert "ghost" not in h.personas and "daily_assistant" in h.personas   # the built-in one is used
    errors = {p["name"]: p["error"] for p in h.problems()}
    assert "unknown model provider 'nowhere'" in errors["ghost"] and "daily_assistant" in errors


def test_mcp_server_failures_are_listed(cfg):
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne()))
    h.mcp.errors["files"] = "command not found"
    assert {"kind": "mcp", "name": "files", "path": "", "error": "command not found", "skipped": True} in h.problems()


def test_home_folder_is_laid_over_the_checkout(tmp_path, monkeypatch):
    home = tmp_path / "home"
    write(home / "config.yaml", "workspace: data\nthresholds: {persona: 0.42}\n")
    monkeypatch.setenv("JEVH_HOME", str(home))
    c = load_config(ROOT / "config.yaml")
    assert c.home == home and c.workspace == home / "data" and c.thresholds["persona"] == 0.42
    assert c.personas_dirs == [home / "personas", ROOT / "personas"]
    assert c.skills_dirs[0] == home / "skills" and ROOT / "skills" in c.skills_dirs


def test_no_home_folder_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("JEVH_HOME", str(tmp_path / "missing"))
    c = load_config(ROOT / "config.yaml")
    assert c.home is None and c.workspace == ROOT / "workspace" and c.personas_dirs == [ROOT / "personas"]
