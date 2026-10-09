"""procs.run: other programs (project tests, failure hooks, project.sh) end on time and never wait on a stdin."""
import os
import subprocess
import sys
import time

import pytest

from jevh import procs

# a child that starts a grandchild holding the same stdout, then both sleep: on Windows, subprocess.run(...,
# timeout=) killed only the child and then waited for the grandchild to let go of the pipe - a minute here
TREE = ("import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open(sys.argv[1], 'w').write(str(g.pid))\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n")


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_timeout_kills_the_whole_tree_and_returns_on_time(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    t = time.time()
    with pytest.raises(subprocess.TimeoutExpired) as e:
        procs.run([sys.executable, "-c", TREE, str(pid_file)], timeout=3)
    assert time.time() - t < 3 + procs.DRAIN + 2
    assert "started" in (e.value.stdout or "")            # the output read before the timeout comes back
    pid = int(pid_file.read_text())
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(pid), "the grandchild outlived the timeout"


def test_the_program_gets_an_empty_stdin_unless_handed_input():
    read = [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"]
    assert procs.run(read, timeout=20).stdout.strip() == "''"
    assert procs.run(read, timeout=20, input="hello").stdout.strip() == "'hello'"


def test_a_shell_command_runs_with_its_exit_code_and_output(tmp_path):
    r = procs.run(f'"{sys.executable}" -c "import sys; print(40 + 2); sys.exit(3)"', shell=True, cwd=tmp_path,
                  timeout=20)
    assert r.returncode == 3 and r.stdout.strip() == "42"
