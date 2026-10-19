"""jevh home: moving your own things out of the checkout (jevh/home.py), on a fake install."""
import pytest
import yaml

from jevh import home as homemod
from jevh.config import load_config
from jevh.personas import load_personas
from jevh.skills import load_roster

from conftest import git

PERSONA = "description: {d}\nsystem_prompt: hi\nmodel: {{provider: lmstudio, model: m}}\n"
SKILL = "---\nname: {n}\ndescription: does {n}\n---\nHow to {n}.\n"


def sh(cwd, *args):
    return git(cwd, *args)   # (skips the test where git isn't installed)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A checkout as shipped, then used: edits, additions, learned skills, settings, data."""
    repo, home = tmp_path / "repo", tmp_path / "home"
    write(repo / "config.yaml", "workspace: workspace\nskills_dirs: [skills]\nthresholds: {persona: 0.5, collab: 0.6}\n"
                                "providers: {lmstudio: {type: openai, base_url: http://x}}\n")
    write(repo / ".gitignore", "workspace/\nconfig.local.yaml\nskills/learned/\nskills/_lessons/*\n!skills/_lessons/shipped.md\n")
    write(repo / "personas" / "coach.yaml", PERSONA.format(d="built-in coach"))
    write(repo / "skills" / "plan" / "SKILL.md", SKILL.format(n="plan"))
    write(repo / "skills" / "write" / "SKILL.md", SKILL.format(n="write"))
    write(repo / "skills" / "_lessons" / "shipped.md", "- a shipped lesson\n")
    sh(tmp_path, "init", "-q", "-b", "main", str(repo))
    sh(repo, "add", "-A")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "shipped")
    # ...then used
    write(repo / "personas" / "coach.yaml", PERSONA.format(d="my coach"))
    write(repo / "personas" / "mine.yaml", PERSONA.format(d="my own persona"))
    write(repo / "skills" / "plan" / "SKILL.md", SKILL.format(n="plan") + "\n## Lessons from past runs\n- learned one\n")
    write(repo / "skills" / "write" / "SKILL.md", SKILL.format(n="write").replace("How to", "My way to"))
    write(repo / "skills" / "_lessons" / "shipped.md", "- a shipped lesson\n- and one I learned\n")
    write(repo / "skills" / "_lessons" / "hermes-thing.md", "- a lesson for a Hermes skill\n")
    write(repo / "skills" / "learned" / "new-trick" / "SKILL.md", SKILL.format(n="new-trick"))
    write(repo / "skills" / "mytool" / "SKILL.md", SKILL.format(n="mytool"))
    write(repo / "config.local.yaml", "models: {fast: {provider: lmstudio, model: small}}\nskills_dirs: [skills, extra]\n")
    write(repo / "workspace" / "memory" / "memory.db", "data")
    cfg_text = (repo / "config.yaml").read_text().replace("collab: 0.6", "collab: 0.7")
    write(repo / "config.yaml", cfg_text)
    write(repo / "jevh_patch.py", "# my code\n")
    monkeypatch.setattr(homemod, "ROOT", repo)
    monkeypatch.setenv("JEVH_CONFIG", str(repo / "config.yaml"))
    monkeypatch.setenv("JEVH_HOME", str(home))
    monkeypatch.delenv("JEVH_NO_LOCAL_CONFIG")
    return repo, home


def test_plan_lists_everything_and_changes_nothing(install):
    repo, home = install
    steps = homemod.plan(home)
    text = homemod.show(steps, home)
    for what in ("config.local.yaml", "settings you changed in config.yaml", "your data (workspace)",
                 "learned skill new-trick", "lesson notes for hermes-thing", "lessons added to shipped.md",
                 "your persona mine", "your skill mytool", "your edits to the persona coach",
                 "lessons learned for the skill plan", "your edits to the skill write", "jevh_patch.py"):
        assert what in text, what
    assert not home.exists() and (repo / "config.local.yaml").exists()


def test_move_leaves_the_checkout_as_shipped_and_everything_still_loads(install):
    repo, home = install
    homemod.move(home)
    # the checkout: only the code file you added is left
    assert sh(repo, "status", "--porcelain").splitlines() == ["?? jevh_patch.py"]
    assert not (repo / "config.local.yaml").exists() and not (repo / "workspace").exists()
    assert (home / "workspace" / "memory" / "memory.db").read_text() == "data"
    conf = yaml.safe_load((home / "config.yaml").read_text())
    assert conf["thresholds"] == {"collab": 0.7} and conf["workspace"] == "workspace"
    assert conf["models"]["fast"]["model"] == "small"
    assert conf["skills_dirs"] == [str(repo / "skills"), str(repo / "extra")]   # still the same folders
    assert (home / "config.local.yaml.old").is_file()
    # the harness sees the same things as before the move
    cfg = load_config()
    assert cfg.home == home and cfg.workspace == home / "workspace" and cfg.thresholds["collab"] == 0.7
    personas = load_personas(cfg.personas_dirs)
    assert personas["coach"].description == "my coach" and "mine" in personas
    roster = load_roster(cfg.skills_dirs, [d / "_lessons" for d in cfg.skills_dirs])
    assert "learned one" in roster["plan"].body and roster["plan"].path.startswith(str(repo))   # built-in, plus notes
    assert "My way to" in roster["write"].body and "new-trick" in roster and "mytool" in roster
    assert (home / "skills" / "_lessons" / "shipped.md").read_text() == "- and one I learned\n"
    assert (repo / "skills" / "_lessons" / "shipped.md").read_text() == "- a shipped lesson\n"
    assert (home / "skills" / "_lessons" / "hermes-thing.md").is_file()
    assert homemod.plan(home)[0]["what"] == "a file you added: jevh_patch.py"   # nothing else left


def test_main_dry_run_by_default(install, capsys):
    repo, home = install
    assert homemod.main([]) == 0
    assert "jevh home move" in capsys.readouterr().out and not home.exists()
