"""Run ARC benchmark jobs one after another (LM Studio holds one model at a time).

    python evals/bench/queue.py ornith-base-dev ornith-harness-dev ...
    python evals/bench/queue.py --list

Each job is arc.py with a model, a mode and a task set; results go to runs/arc1_<mode>_<tag>.jsonl
and a finished task is never re-run (arc.py resumes), so a stopped queue can simply be restarted.
One line per task result on stdout.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {"ornith": "ornith-1.5-35b-a3b", "gemma": "google/gemma-4-12b-qat", "qwen": None}   # None: persona's
LIMITS = {"ornith": 1200, "gemma": 1200, "qwen": 1800}                                       # seconds per task
SETS = {  # dev: seed 0 (the harness is tuned on these, with ornith); held: unseen tasks
    "dev": ["--n", "10", "--seed", "0"],
    "held": ["--ids", "c48954c1", "e7639916", "15663ba9", "e66aafb8", "759f3fd3", "351d6448", "a3f84088",
             "9b365c51", "195ba7dc", "d017b73f"],
}


def job_args(job: str) -> tuple[list[str], str]:
    """'ornith-harness-dev[-v2]' -> arc.py args, tag."""
    model, mode, which, *ver = job.split("-")
    tag = f"{model}_{which}" + (f"_{ver[0]}" if ver else "")
    args = ["--mode", mode, "--tag", tag, "--limit", str(LIMITS[model]), *SETS[which]]
    if MODELS[model]:
        args += ["--model", MODELS[model]]
    if mode == "harness":
        args += ["--thinking-watch"]
    return args, tag


def main() -> None:
    if "--list" in sys.argv:
        for m in MODELS:
            for mode in ("base", "harness"):
                for s in SETS:
                    print(f"{m}-{mode}-{s}")
        return
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for job in sys.argv[1:]:
        args, tag = job_args(job)
        print(f"=== JOB {job}: arc.py {' '.join(args)}", flush=True)
        p = subprocess.Popen([sys.executable, "-u", str(HERE / "arc.py"), *args], cwd=HERE.parent.parent,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", env=env)
        with open(HERE / "runs" / f"queue_{tag}_{args[1]}.log", "a", encoding="utf-8") as log:
            for line in p.stdout:
                log.write(line)
                log.flush()
                if line.startswith("[") or "Traceback" in line or "STOP" in line:
                    print(f"{job}: {line.strip()[:200]}", flush=True)
        p.wait()
        print(f"=== END {job} (exit {p.returncode})", flush=True)
    print("=== QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
