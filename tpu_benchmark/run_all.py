"""
Run all benchmark scripts with progress bar and per-test logging.

Usage:
    cd /mnt2/users/tilelanguser-xxw/tilelang-tpu
    python tpu_benchmark/run_all.py
    python tpu_benchmark/run_all.py --ops swiglu rope
    python tpu_benchmark/run_all.py --sizes 64 128
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

OP_ORDER = [
    ("swiglu", "swiglu/bench_swiglu_*.py"),
    ("rope", "rope/bench_rope_*.py"),
    ("add", "elementwise_add/bench_add_*.py"),
]

BAR_WIDTH = 40


def collect_scripts(ops=None, sizes=None):
    scripts = []
    for op, pattern in OP_ORDER:
        if ops and op not in ops:
            continue
        for f in sorted(glob.glob(os.path.join(SCRIPT_DIR, pattern))):
            if sizes:
                m = re.search(r'(\d+)x(\d+)\.py$', f)
                if m and int(m.group(1)) not in sizes:
                    continue
            scripts.append(f)
    return scripts


def progress_bar(cur, total, passed, failed, elapsed):
    filled = int(cur * BAR_WIDTH / total) if total else 0
    bar = '#' * filled + '-' * (BAR_WIDTH - filled)
    pct = cur * 100 // total if total else 0
    eta = ""
    if cur > 0:
        avg = elapsed / cur
        remain = avg * (total - cur)
        m, s = divmod(int(remain), 60)
        eta = f"  ETA {m}m{s:02d}s"
    sys.stdout.write(f"\r  [{bar}] {pct:3d}% ({cur}/{total})  "
                     f"pass={passed} fail={failed}{eta}   ")
    sys.stdout.flush()


def parse_output(text):
    """Extract key metrics from benchmark output."""
    results = {}
    for m in re.finditer(
        r'\[(\w+)\]\s+correct=(\w+)\s+max_diff=([\d.]+)\s+avg_diff=([\d.]+)', text
    ):
        backend, correct, max_diff, avg_diff = m.groups()
        results[backend] = {
            "correct": correct == "True",
            "max_diff": float(max_diff),
            "avg_diff": float(avg_diff),
        }
    for m in re.finditer(r'Average execution time:\s+([\d.]+)\s+ms', text):
        for backend in ["tilelang", "PPL"]:
            if backend in results and "avg_ms" not in results[backend]:
                results[backend]["avg_ms"] = float(m.group(1))
                break
    return results


def run_one(script, logfile, timeout=300):
    try:
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True, timeout=timeout,
            cwd=REPO_ROOT,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        full = output + "\n" + stderr
        with open(logfile, "w") as f:
            f.write(full)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        with open(logfile, "w") as f:
            f.write("TIMEOUT\n")
        return "TIMEOUT", {}

    metrics = parse_output(full)
    if rc != 0 and not metrics:
        return "CRASH", {}

    all_correct = all(v.get("correct", False) for v in metrics.values())
    return "PASS" if (all_correct or metrics) else "FAIL", metrics


def fmt_metrics(metrics):
    parts = []
    for backend in ["tilelang", "PPL"]:
        if backend not in metrics:
            continue
        m = metrics[backend]
        ok = "Y" if m.get("correct") else "N"
        ms = f"{m['avg_ms']:.3f}ms" if "avg_ms" in m else "?"
        parts.append(f"{backend}={ok}/{ms}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    scripts = collect_scripts(args.ops, args.sizes)
    if not scripts:
        print("No benchmark scripts found.")
        return

    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"run_all_{ts}.log")

    total = len(scripts)
    passed = failed = 0
    summary_lines = []

    def log(line):
        summary_lines.append(line)
        with open(summary_path, "a") as f:
            f.write(line + "\n")

    header = (f"TPU Benchmark Runner  {datetime.now()}\n"
              f"Total: {total} scripts\n" + "=" * 70)
    print(header)
    log(header)

    t0 = time.time()

    for i, script in enumerate(scripts):
        name = os.path.basename(script).replace(".py", "")
        logfile = os.path.join(log_dir, f"{name}_{ts}.log")

        progress_bar(i, total, passed, failed, time.time() - t0)

        status, metrics = run_one(script, logfile, args.timeout)

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        detail = fmt_metrics(metrics) if metrics else status
        line = f"[{i+1}/{total}] {status:<7} {name:<40} {detail}"
        sys.stdout.write("\r" + " " * 100 + "\r")
        print(line)
        log(line)

        progress_bar(i + 1, total, passed, failed, time.time() - t0)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)

    sys.stdout.write("\r" + " " * 100 + "\r")
    footer = (f"\n{'=' * 70}\n"
              f"Done: {total} total, {passed} passed, {failed} failed  "
              f"({m}m{s:02d}s)\n"
              f"Summary: {summary_path}\n"
              f"Logs:    {log_dir}/*_{ts}.log")
    print(footer)
    log(footer)


if __name__ == "__main__":
    main()
