"""The failure log: harness signals, the final() gate, Jev's check, report_failure() and hooks."""
import json
import sys
from pathlib import Path

from jevh.failures import FailureLog, Hook, as_failures, main
from jevh.jev import Jev
from jevh.orchestrator import Harness

from conftest import FakeSystemOne, ScriptedLLM, choice_ans, noul_ans, py


def rules(gave_up=0.05, missed=0.05, **extra):
    def r(k, q, s):
        if k == "persona":
            return choice_ans("daily_assistant", 0.9, q["criteria"])
        if k == "gave_up":
            return noul_ans(gave_up)
        if k == "missed":
            return noul_ans(missed)
        if k in extra:
            return noul_ans(extra[k])
    return r


def harness(cfg, replies, events=None, failures=None, **scores):
    if failures is not None:
        cfg.raw = {**cfg.raw, "failures": failures}
    llm = ScriptedLLM(replies)
    h = Harness(cfg, jev=Jev(cfg, FakeSystemOne(rules(**scores))), llm_factory=lambda p: llm,
                on_event=(lambda k, d: events.append((k, d))) if events is not None else None)
    h.learning = False
    return h, llm


def test_log_records_lists_counts_triages_and_exports(tmp_path):
    log = FailureLog(tmp_path / "f.db")
    a = log.record(source="harness", kind="crash", summary="boom", session="s1", persona="dev", scores={"x": 1})
    log.record(source="gate", kind="gate_bounce", summary="placeholder text", session="s1", persona="dev")
    log.record(source="model", kind="reported", summary="web.ask timed out", session="s2", persona="coach")
    assert [f["kind"] for f in log.list()] == ["reported", "gate_bounce", "crash"]   # newest first
    assert [f["kind"] for f in log.list(session="s1", source="harness")] == ["crash"]
    assert log.list(kind="crash")[0]["scores"] == {"x": 1}
    assert log.counts()["by_persona"] == {"dev": 2, "coach": 1}
    assert log.set_status(a["id"], "fixed", "handled in the worker") and not log.set_status(999, "fixed")
    assert log.counts()["by_kind"] == {"gate_bounce": 1, "reported": 1}    # counts are of open ones
    assert log.list(status="fixed")[0]["note"] == "handled in the worker"
    n = log.export(tmp_path / "out.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert n == 3 and [r["kind"] for r in rows] == ["crash", "gate_bounce", "reported"]   # oldest first


def test_step_limit_is_logged_with_what_is_needed_to_study_it(cfg):
    events = []
    h, _ = harness(cfg, [py("print('working on it')"), py("print('still going')")], events)
    h.personas["daily_assistant"].max_steps = 2
    h.turn("write the report")
    [f] = h.failures.list()
    assert (f["source"], f["kind"], f["persona"], f["turn"], f["task"]) == ("harness", "step_limit",
                                                                             "daily_assistant", 1, 1)
    assert f["request"] == "write the report" and "step limit" in f["answer"]
    assert "still going" in f["work_log"] and Path(f["workdir"], "checkpoint.json").exists()
    assert f["model"] == "ornith-1.5-35b-a3b" and f["session"] == h.session   # the model it ran on (fast tier)
    assert [d["failure"]["kind"] for k, d in events if k == "failure"] == ["step_limit"]


def test_a_crash_is_logged(cfg):
    h, llm = harness(cfg, [])

    def boom(messages):
        raise ConnectionError("model server went away")
    llm.chat = boom
    h.turn("hi")
    [f] = h.failures.list()
    assert f["kind"] == "crash" and "model server went away" in f["answer"]


def test_gate_bounces_are_logged_with_the_answer_they_sent_back(cfg):
    cfg.raw = {**cfg.raw, "verify": {"enabled": True, "max_bounces": 2, "render_html": False}}
    h, _ = harness(cfg, [py("p = save_artifact('page.html', '<h1>[Your Company]</h1>')\nfinal('Built it.', [p])"),
                         py("p = save_artifact('page.html', '<h1>Kettle and Co</h1><p>Warm mugs.</p>')\n"
                            "final('Built it.', [p])")],
                   claims=0.05, delivers=0.95, honest=0.1, accurate=0.95)
    assert h.turn("build a landing page").answer == "Built it."
    [f] = h.failures.list()
    assert (f["source"], f["kind"]) == ("gate", "gate_bounce")
    assert "[Your Company]" in f["detail"] and f["answer"] == "Built it."


def test_jev_check_flags_an_answer_that_gave_up(cfg):
    h, _ = harness(cfg, ["I couldn't find any flights for those dates."], gave_up=0.95)
    h.turn("find me a flight to Lisbon on the 3rd")
    [f] = h.failures.list()
    assert (f["source"], f["kind"]) == ("jev", "gave_up") and f["scores"]["gave_up"] == 0.95
    assert f["answer"].startswith("I couldn't find")


def test_jev_check_passes_a_good_answer_and_can_be_switched_off(cfg):
    h, _ = harness(cfg, ["Hello Jed!"])
    h.turn("hi")
    assert h.failures.list() == []
    h2, _ = harness(cfg, ["Nothing found."], failures={"jev_check": False}, gave_up=0.99)
    h2.turn("find it")
    assert h2.failures.list() == []


def test_a_worker_reports_a_failure_and_carries_on(cfg):
    h, _ = harness(cfg, [py("print(report_failure('web.ask returned an empty list for a plain query', "
                            "'query: best rest times'))"), py("final('done anyway')")])
    r = h.turn("research rest times")
    assert r.answer == "done anyway"
    [f] = h.failures.list()
    assert (f["source"], f["kind"], f["persona"]) == ("model", "reported", "daily_assistant")
    assert f["summary"].startswith("web.ask returned") and f["detail"] == "query: best rest times"
    assert f["task"] == 1 and f["request"] == "research rest times"


def test_command_hooks_flag_failures_from_the_event(cfg, tmp_path):
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "e = json.load(sys.stdin)\n"
        "if 'Lisbon' in e['data']['task']['text'] and 'CODE' not in e['data']['work_log']:\n"
        "    print(json.dumps({'kind': 'no_tools_used', 'summary': 'answered a travel question from memory',\n"
        "                      'detail': e['persona']}))\n", encoding="utf-8")
    plain = tmp_path / "plain.py"   # exit 2 + text is a failure too
    plain.write_text("import sys\nprint('the reply is too short')\nsys.exit(2)\n", encoding="utf-8")
    broken = tmp_path / "broken.py"
    broken.write_text("raise SystemExit(3)\n", encoding="utf-8")
    events = []
    exe = f'"{sys.executable}"'
    h, _ = harness(cfg, ["Flights to Lisbon are about $400."], events, failures={"hooks": [
        {"on": ["done"], "command": f'{exe} "{script}"'},
        {"name": "short_reply", "on": "done", "command": f'{exe} "{plain}"'},
        {"on": ["done"], "command": f'{exe} "{broken}"'}]})
    h.turn("how much is a flight to Lisbon?")
    got = {f["kind"]: f for f in h.failures.list()}
    assert set(got) == {"no_tools_used", "short_reply"}
    assert got["no_tools_used"]["source"] == "hook" and got["no_tools_used"]["detail"] == "daily_assistant"
    assert got["short_reply"]["summary"] == "the reply is too short"
    assert got["no_tools_used"]["answer"] == "Flights to Lisbon are about $400."
    errors = [d for k, d in events if k == "hook_error"]
    assert len(errors) == 1 and "exit 3" in errors[0]["error"]   # a broken hook is shown, not logged


def test_python_hooks_and_what_a_hook_may_return(tmp_path):
    (tmp_path / "myhooks.py").write_text(
        "def check(event):\n"
        "    if 'Traceback' in event['data'].get('output', ''):\n"
        "        return {'failures': ['a traceback', {'kind': 'crashy', 'summary': 'again', 'scores': {'p': 1}}]}\n",
        encoding="utf-8")
    hook = Hook({"on": ["output"], "python": "myhooks:check"}, tmp_path)
    assert hook.wants("output") and not hook.wants("done")
    assert hook({"data": {"output": "ok"}}) == []
    assert hook({"data": {"output": "Traceback ..."}}) == [
        {"summary": "a traceback"}, {"kind": "crashy", "summary": "again", "scores": {"p": 1}}]
    assert as_failures(None) == [] and as_failures({"ok": True}) == [] and as_failures("") == []
    assert as_failures({"failure": "x"}) == [{"summary": "x"}]
    assert as_failures({"kind": "k"}) == [{"kind": "k", "summary": "k"}]


def test_logging_can_be_switched_off(cfg):
    h, _ = harness(cfg, [py("print(report_failure('x'))"), py("final('ok')")], failures={"enabled": False},
                   gave_up=0.99)
    assert h.turn("go").answer == "ok"
    assert h.failures.list() == []


def test_command_line_lists_exports_and_triages(cfg, tmp_path, capsys, monkeypatch):
    log = FailureLog(cfg.workspace / "failures.db")
    fid = log.record(source="harness", kind="stalled", summary="went in circles", persona="dev")["id"]
    monkeypatch.setattr("jevh.config.load_config", lambda path=None: cfg)
    assert main([]) == 0 and "went in circles" in capsys.readouterr().out
    assert main(["status", str(fid), "reviewed", "seen it"]) == 0
    assert log.list()[0]["status"] == "reviewed"
    assert main(["export", str(tmp_path / "x.jsonl")]) == 0
    assert json.loads((tmp_path / "x.jsonl").read_text(encoding="utf-8"))["kind"] == "stalled"


def test_a_missed_failure_is_added_from_the_task_folder(cfg, tmp_path, capsys, monkeypatch):
    task = tmp_path / "sessions" / "20260930-120314-a809" / "u1t1-high_end_dev"
    task.mkdir(parents=True)
    final = {"id": "c2", "type": "function", "function": {"name": "final", "arguments": json.dumps(
        {"answer": "CSV of the 8 planets", "artifacts": ["planets.csv"]})}}
    (task / "checkpoint.json").write_text(json.dumps({"persona": "high_end_dev", "task_text": "Give me planets.csv",
        "messages": [{"role": "user", "content": "Give me planets.csv"},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                         "name": "python", "arguments": json.dumps({"code": "save_artifact('planets.csv', 'planets.csv')"})}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "artifacts/planets.csv"},
                     {"role": "assistant", "content": "", "tool_calls": [final]}]}), encoding="utf-8")
    monkeypatch.setattr("jevh.config.load_config", lambda path=None: cfg)
    assert main(["add", "bad_artifact", "handed over the name, not the file", "--workdir", str(task),
                 "--model", "ornith"]) == 0
    f = FailureLog(cfg.workspace / "failures.db").list()[0]
    assert (f["source"], f["kind"], f["persona"], f["model"], f["session"]) == \
        ("user", "bad_artifact", "high_end_dev", "ornith", "20260930-120314-a809")
    assert f["request"] == "Give me planets.csv" and f["answer"] == "CSV of the 8 planets"
    assert "save_artifact('planets.csv', 'planets.csv')" in f["work_log"]
    assert main(["add", "x", "no request given"]) == 1
