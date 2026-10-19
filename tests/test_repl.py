import time
import pytest

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


def test_a_secret_in_a_cells_output_is_cut_but_the_code_still_holds_it(tmp_path):
    with make(tmp_path) as r:
        r.run("page = 'Your Client ID 1-abc.apps.googleusercontent.com Your Client secret GOCSPX-AbCdEfGhIjKlMnOpQrStUv_12'")
        shown = r.run("print(page)\npage").output          # printed, and echoed as the last expression
        assert "GOCSPX" not in shown and shown.count("[secret hidden]") == 2
        assert "1-abc.apps.googleusercontent.com" in shown   # a Client ID is not a secret
        assert r.run("import re\nm = re.search(r'GOCSPX-[A-Za-z0-9_-]+', page)\nlen(m.group(0))").output.strip() == "32"
        out = r.run("print('xoxb-' + '1234567890-abcdefghijklmnop', 'ghp_' + 'a' * 36, 'AKIA' + 'B' * 16)").output
        assert "xoxb" not in out and "ghp_" not in out and "AKIA" not in out and out.count("[secret hidden]") == 3
        assert r.run("print('a normal line, sk-learn and AKIAN are fine')").output.strip() == "a normal line, sk-learn and AKIAN are fine"
        assert r.run("print('-----BEGIN RSA PRIVATE KEY-----\\nMIIB\\n-----END RSA PRIVATE KEY-----')").output.strip() == "[secret hidden]"


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
        assert r.run("keep").error  # state was lost, but the REPL works (the worker restores its last snapshot)


def test_subprocess_stdout_cannot_break_protocol(tmp_path):
    with make(tmp_path) as r:
        r.run("import os; os.system('echo noise')")
        assert r.run("2 * 3").output.strip() == "6"


def test_input_is_blocked(tmp_path):
    with make(tmp_path) as r:
        assert r.run("input('?')").error


def test_a_subprocess_reading_stdin_gets_an_end_of_file_not_the_protocol_pipe(tmp_path):
    """A child started without stdin= inherited the REPL's stdin, which is the host's protocol pipe: one that read it (a
    prompt, a test calling input()) waited there until the step timed out, or swallowed the host's answer to a
    callback, and the REPL then waited for that answer forever."""
    with make(tmp_path, llm_query=lambda p, data=None: p.upper()) as r:
        t = time.time()
        res = r.run("import subprocess, sys\n"
                    "child = subprocess.Popen([sys.executable, '-c', 'import sys; print(repr(sys.stdin.read()))'],"
                    " stdout=subprocess.PIPE, text=True)\n"
                    "print(llm_query('still here'))\n"
                    "print('child read', child.communicate(timeout=10)[0].strip())")
        assert not res.error, res.output
        assert "STILL HERE" in res.output and "child read ''" in res.output
        assert time.time() - t < 10


def test_a_lone_surrogate_in_the_output_does_not_kill_the_repl(tmp_path):
    """Half an emoji (a JSON string cut mid-pair) printed by the worker could not be written to the host as UTF-8: the
    REPL died, and every variable with it."""
    with make(tmp_path) as r:
        r.run("keep = 7")
        res = r.run("print('half an emoji:', chr(0xD83D))")
        assert not res.error and "half an emoji:" in res.output
        assert r.run("keep").output.strip() == "7"   # the same REPL, its variables intact


def test_a_variable_too_big_to_snapshot_is_skipped_without_losing_the_rest(tmp_path):
    """The snapshot after every step pickled each variable in full before dropping the ones over 16 MB: a large one
    held a second copy in memory and could outlast the snapshot's time limit, which restarts the REPL and loses every
    variable. It now stops pickling a variable once it passes the cap.

    What is checked is the work, not the clock: how far each pickle grew, counted in the REPL's own capped writer. This
    test used to assert the snapshot took < 0.4 s (0.05 s capped, 0.67 s uncapped), and failed on a machine with a local
    model on every core (issue #7); on one machine the same snapshot, repeated, took 0.05 s to 2.9 s."""
    with Repl(tmp_path / "w", tmp_path / "a", timeout=20) as r:
        r.run("small = [3, 1, 4]\nbig = [str(i) * 12 for i in range(3_000_000)]")   # ~150 MB as a pickle
        r.run("import sys\n"   # (names starting with _ and modules are never snapshotted)
              "_srv = sys.modules['__main__']\n"   # the REPL server: jevh/repl/server.py
              "_grew = []\n"
              "def _counting(self, data, _write=_srv._Capped.write):\n"
              "    _grew.append((self.cap, len(self.buf) + memoryview(data).nbytes))\n"
              "    return _write(self, data)\n"
              "_srv._Capped.write = _counting")
        info = r.snapshot(tmp_path / "state.pkl")
        assert info["saved"] == ["small"] and "big" in info["skipped"], info
        peak, stopped, cap = eval(r.run("max((n for _, n in _grew), default=0), any(n > c for c, n in _grew), "
                                        "_srv.MAX_VAR_BYTES")
                                  .output.strip())
        assert stopped                       # big's pickle reached the cap and was stopped there...
        assert peak <= cap + 1024 * 1024     # ...one write past it, never the whole ~150 MB first
        assert r.run("len(big), small").output.strip() == "(3000000, [3, 1, 4])"   # nothing was lost


def test_output_truncated(tmp_path):
    with make(tmp_path) as r:
        out = r.run("print('x' * 5000)").output
        assert len(out) < 400 and "omitted" in out


def test_truncate_keeps_head_and_tail():
    s = "A" * 100 + "B" * 100
    t = truncate(s, 50)
    assert t.startswith("A") and t.endswith("B")


# ---------------------------------------------------------------- look_at(): a file shown to the worker's own model (#2)
def _looker(seen, reply="it says TOTAL 42.17"):
    def look(path, question):
        from PIL import Image
        with Image.open(path) as im:
            seen.append({"path": path, "size": im.size, "question": question})
        return reply
    return look


def test_look_at_shows_a_photo_upright_and_scaled_down_to_the_worker_model(tmp_path):
    """Issue #2: the worker found no way to look at a photo or a scan, decided it had no vision and installed OCR. A phone
    photo (4000x3000, stored on its side with an EXIF tag) goes to the model upright and at most LOOK_SIDE px."""
    Image = pytest.importorskip("PIL.Image")
    photo = tmp_path / "receipt.jpg"
    im = Image.new("RGB", (4000, 3000), "white")
    exif = im.getexif()
    exif[0x0112] = 6                       # "rotate 90 degrees to show": the camera held upright
    im.save(photo, exif=exif)
    seen = []
    with Repl(tmp_path / "w", tmp_path / "a", {"look": _looker(seen)}, timeout=30) as r:
        res = r.run(f"look_at(r'{photo}', 'what is the total?')")
        assert not res.error and "TOTAL 42.17" in res.output
    assert len(seen) == 1 and seen[0]["size"] == (1200, 1600)   # turned upright, long side 1600
    assert seen[0]["question"].endswith("what is the total?") and "receipt.jpg" in seen[0]["question"]


def test_look_at_renders_each_pdf_page_and_labels_the_answers(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    pytest.importorskip("PIL")
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.save(tmp_path / "scan.pdf")
    seen = []
    with Repl(tmp_path / "w", tmp_path / "a", {"look": _looker(seen)}, timeout=30) as r:
        out = r.run(f"print(look_at(r'{tmp_path / 'scan.pdf'}'))").output
        assert "[Page 1 of 3]" in out and "[Page 3 of 3]" in out and out.count("TOTAL 42.17") == 3
        assert "Transcribe all the text" in seen[0]["question"]               # no question: read it all
        out = r.run(f"print(look_at(r'{tmp_path / 'scan.pdf'}', 'the date?', pages=[2]))").output
        assert "[Page 2 of 3]" in out and "Page 1" not in out
        assert "numbered 1-3" in r.run(f"look_at(r'{tmp_path / 'scan.pdf'}', pages=[9])").output
    assert max(seen[0]["size"]) == 1600


def test_look_at_says_plainly_when_the_model_takes_no_images(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (50, 50)).save(tmp_path / "a.png")
    text_only = _looker([], "(could not look at the image: 400 image input is not supported - this model may not take images)")
    with Repl(tmp_path / "w", tmp_path / "a", {"look": text_only}, timeout=30) as r:
        out = r.run(f"print(look_at(r'{tmp_path / 'a.png'}'))").output
    assert "can't see images" in out and "before installing OCR" in out


def test_look_at_refuses_what_is_not_a_picture_and_a_missing_file(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    with Repl(tmp_path / "w", tmp_path / "a", {"look": _looker([])}, timeout=30) as r:
        assert "read text files as text" in r.run(f"look_at(r'{tmp_path / 'data.csv'}')").output
        assert "FileNotFoundError" in r.run("look_at('nope.png')").output


def test_a_missing_ocr_package_points_the_worker_at_look_at(tmp_path):
    with Repl(tmp_path / "w", tmp_path / "a", timeout=30) as r:
        out = r.run("import rapidocr_not_here_either\n").output
        assert "look_at" not in out                                          # any other missing module: no hint
        out = r.run("import sys\nsys.modules['easyocr'] = None\nimport easyocr").output
        assert "ModuleNotFoundError" in out or "ImportError" in out
        assert "look_at(path, question)" in out
