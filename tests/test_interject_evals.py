"""The interjection evals' own code, offline: the LM Studio log reader (on lines copied from a real log) and the
classifier eval's scoring (on the raw scores committed beside it, so the numbers the docs quote stay reproducible)."""
import json
import sys

from jevh.config import ROOT, load_config

sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "evals" / "bench"))

import interject_cases as ic  # noqa: E402
import lmstudio_log as lml  # noqa: E402

# lines as LM Studio's llama.cpp wrote them (2026-10-03): a request cut during its prompt, its retry on the other slot,
# then a request that found 88% of its prompt already cached
LOG = """\
[2026-10-03 23:17:09][DEBUG] 257.10.301.221 I slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = 624648907
[2026-10-03 23:17:09][DEBUG] 257.10.302.115 I slot launch_slot_: id  1 | task 9166 | processing task, is_child = 0
[2026-10-03 23:17:16][INFO][LM STUDIO SERVER] Client disconnected. Stopping generation... (If the model is busy processing the prompt, it will finish first.)
[2026-10-03 23:17:16][DEBUG] 257.17.449.007 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = 626452410
[2026-10-03 23:17:16][DEBUG] 257.17.450.981 W srv          stop: cancel task, id_task = 9166
[2026-10-03 23:17:16][DEBUG] 257.17.451.002 I slot launch_slot_: id  0 | task 9167 | processing task, is_child = 0
[2026-10-03 23:17:19][DEBUG] 257.20.115.334 I slot      release: id  1 | task 9166 | stop processing: n_tokens = 533, truncated = 0
[2026-10-03 23:17:36][DEBUG] 257.37.300.001 I slot print_timing: id  0 | task 9167 | prompt eval time =    4527.68 ms /  2031 tokens (    2.23 ms per token,   448.57 tokens per second)
[2026-10-03 23:17:36][DEBUG] 257.37.300.011 I slot print_timing: id  0 | task 9167 |        eval time =   16200.00 ms /   511 tokens (   31.70 ms per token,    31.54 tokens per second)
[2026-10-03 23:17:36][DEBUG] 257.37.300.020 I slot      release: id  0 | task 9167 | stop processing: n_tokens = 2541, truncated = 0
[2026-10-03 23:17:36][DEBUG] 257.37.301.100 I slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.881 (> 0.100 thold), f_keep = 0.912
[2026-10-03 23:17:36][DEBUG] 257.37.301.200 I slot launch_slot_: id  0 | task 9684 | processing task, is_child = 0
[2026-10-03 23:17:40][DEBUG] 257.41.000.001 I slot print_timing: id  0 | task 9684 | prompt eval time =    1772.00 ms /   310 tokens (    5.72 ms per token,   174.94 tokens per second)
[2026-10-03 23:17:40][DEBUG] 257.41.000.002 I slot print_timing: id  0 | task 9684 |        eval time =    2000.00 ms /    64 tokens (   31.25 ms per token,    32.00 tokens per second)
[2026-10-03 23:17:40][DEBUG] 257.41.000.003 I slot      release: id  0 | task 9684 | stop processing: n_tokens = 2605, truncated = 0
"""


def test_the_log_reader_tells_what_the_server_did_with_each_request():
    cut, retry, cached = lml.parse(LOG.splitlines())
    assert (cut["slot"], cut["task"], cut["how"], cut["cancelled"], cut["processed"]) == (1, 9166, "LRU", True, None)
    assert (retry["slot"], retry["how"], retry["sim"], retry["processed"], retry["generated"]) == (0, "LRU", None, 2031, 511)
    assert lml.prompt_tokens(retry) == 2541 - 511 and retry["gen_tps"] == 31.54
    assert (cached["how"], cached["sim"], cached["keep"], cached["processed"]) == ("LCP", 0.881, 0.912, 310)
    assert lml.prompt_tokens(cached) == 2605 - 64
    assert 1 - cached["processed"] / lml.prompt_tokens(cached) > 0.85        # (what the docs call "mostly cached")


def test_the_log_reader_ignores_other_lines():
    assert lml.parse(["[2026-10-03 22:10:13][DEBUG] Received request: GET to /api/v1/models", "", "junk"]) == []


def test_the_classifier_eval_scores_its_own_committed_raw_scores():
    """The numbers in config.yaml and the docs come from these files: --offline must still give them."""
    status_at = load_config(ROOT / "config.yaml").t("interject_status")
    raw = json.loads(ic.RAW.read_text(encoding="utf-8"))
    sizes = {"dev": len(ic.DEV), "held": len(ic.HELD), "fresh": len(ic.FRESH), "real": len(ic.REAL),
             "real_fresh": len(ic.REAL_FRESH)}
    right = total = wrongly_answered = 0
    for name, n in sizes.items():
        assert len(raw[name]) == n, name                                   # the raw scores belong to these cases
        e = ic.evaluate(raw[name], status_at)
        # the costly mistakes: a correction, another request or a stop answered on the side instead of added to the prompt
        assert e["others_answered"] == 0, name
        assert e["accuracy"] >= 0.85, name
        right, total = right + e["right"], total + e["total"]
        wrongly_answered += e["corrections_answered"]
    assert wrongly_answered == 1                                           # "the crash only happens with a zero-byte file"
    assert (right, total) == (209, 214)                                    # the figures quoted in config.yaml and the README


def test_the_eval_wants_a_question_answered_and_everything_else_added():
    assert [ic.want(k) for k in ic.KINDS] == ["correction", "status", "correction", "correction"]
    rows = [{"label": "correction", "probs": {"correction": 0.1, "status": 0.8, "separate": 0.05, "stop": 0.05}},
            {"label": "stop", "probs": {"correction": 0.1, "status": 0.8, "separate": 0.05, "stop": 0.05}},
            {"label": "status", "probs": {"correction": 0.8, "status": 0.1, "separate": 0.05, "stop": 0.05}},
            {"label": "status", "probs": {"correction": 0.1, "status": 0.8, "separate": 0.05, "stop": 0.05}}]
    e = ic.evaluate(rows, 0.6)
    assert (e["right"], e["corrections_answered"], e["others_answered"], e["questions_added"]) == (1, 1, 1, 1)


def test_the_eval_cases_are_labelled_with_the_four_kinds_and_balanced():
    for name, cases in (("dev", ic.DEV), ("held", ic.HELD), ("fresh", ic.FRESH)):
        labels = [c[3] for c in cases]
        assert set(labels) == set(ic.KINDS), name
        assert all(c[2].strip() for c in cases), name
        assert min(labels.count(k) for k in ic.KINDS) >= 10, name
