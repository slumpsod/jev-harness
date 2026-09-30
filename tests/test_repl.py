import time

from jevh.repl.client import Repl, truncate


def make(tmp_path, **handlers):
    return Repl(tmp_path / "w", tmp_path / "a", handlers, timeout=3, output_cap=200)


def test_state_persists_and_last_expression_echoes(tmp_path):
    with make(tmp_path) as r:
        assert r.run("x = 20").output == ""
        assert r.run("x + 1").output.strip() == "21"


def test_callbacks_reach_host_and_errors_surface(tmp_path):
    with make(tmp_path, llm_query=lambda p, data=None: p.upper()) as r:
        assert r.run("llm_query('hi')").output.strip() == "'HI'"
        res = r.run("request('x', 'y')")
        assert res.error and "not available" in res.output


def test_exceptions_are_reported(tmp_path):
    with make(tmp_path) as r:
        res = r.run("1/0")
        assert res.error and "ZeroDivisionError" in res.output


def test_final_and_artifacts(tmp_path):
    with make(tmp_path) as r:
        res = r.run("p = save_artifact('note.txt', 'hello')\nfinal('ok', [p])")
        assert res.final["answer"] == "ok"
        assert (tmp_path / "a" / "note.txt").read_text() == "hello"


def test_save_artifact_given_a_file_name_as_data_saves_that_file(tmp_path):
    with make(tmp_path) as r:   # live: save_artifact('planets.csv', 'planets.csv') saved the name as the CSV
        r.run("open('planets.csv', 'w').write('name,order\\nMercury,1\\n')")
        r.run("p = save_artifact('planets.csv', 'planets.csv')\nfinal('ok', ['planets.csv'])")
        assert (tmp_path / "a" / "planets.csv").read_text() == "name,order\nMercury,1\n"
        r.run("save_artifact('hello.txt', 'no such file.txt')")   # not a file: still text
        assert (tmp_path / "a" / "hello.txt").read_text() == "no such file.txt"


def test_timeout_restarts_quickly(tmp_path):
    with make(tmp_path) as r:
        r.run("keep = 1")
        t = time.time()
        res = r.run("while True: pass")
        assert res.timed_out and time.time() - t < 6
        assert r.run("keep").error  # state was lost, but the REPL works


def test_subprocess_stdout_cannot_break_protocol(tmp_path):
    with make(tmp_path) as r:
        r.run("import os; os.system('echo noise')")
        assert r.run("2 * 3").output.strip() == "6"


def test_input_is_blocked(tmp_path):
    with make(tmp_path) as r:
        assert r.run("input('?')").error


def test_output_truncated(tmp_path):
    with make(tmp_path) as r:
        out = r.run("print('x' * 5000)").output
        assert len(out) < 400 and "omitted" in out


def test_truncate_keeps_head_and_tail():
    s = "A" * 100 + "B" * 100
    t = truncate(s, 50)
    assert t.startswith("A") and t.endswith("B")
