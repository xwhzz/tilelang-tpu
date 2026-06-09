"""Run fixed optimized RMSNorm benchmark scripts."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(BENCHMARK_ROOT)
RESULTS_DIR = os.path.join(BENCHMARK_ROOT, "results", "rmsnorm_opt")

CONFIGS = {
    "bf16_64x64": {"variant": "keep_square_b32", "script": "bench_rmsnorm_bf16_64x64.py"},
    "bf16_128x128": {"variant": "keep_square_b128", "script": "bench_rmsnorm_bf16_128x128.py"},
    "bf16_256x256": {"variant": "keep_square_b128", "script": "bench_rmsnorm_bf16_256x256.py"},
    "bf16_512x512": {"variant": "keep_square_b256", "script": "bench_rmsnorm_bf16_512x512.py"},
    "bf16_1024x1024": {
        "variant": "keep_square_b512",
        "script": "bench_rmsnorm_bf16_1024x1024.py",
    },
    "fp16_64x64": {"variant": "reload_b64", "script": "bench_rmsnorm_fp16_64x64.py"},
    "fp16_128x128": {"variant": "reload_b128", "script": "bench_rmsnorm_fp16_128x128.py"},
    "fp16_256x256": {"variant": "keep_square_b256", "script": "bench_rmsnorm_fp16_256x256.py"},
    "fp16_512x512": {"variant": "keep_square_b512", "script": "bench_rmsnorm_fp16_512x512.py"},
    "fp16_1024x1024": {
        "variant": "keep_square_b256",
        "script": "bench_rmsnorm_fp16_1024x1024.py",
    },
    "fp32_64x64": {"variant": "reload_b32", "script": "bench_rmsnorm_fp32_64x64.py"},
    "fp32_128x128": {"variant": "keep_square_b128", "script": "bench_rmsnorm_fp32_128x128.py"},
    "fp32_256x256": {"variant": "reload_b256", "script": "bench_rmsnorm_fp32_256x256.py"},
    "fp32_512x512": {"variant": "keep_square_b512", "script": "bench_rmsnorm_fp32_512x512.py"},
    "fp32_1024x1024": {
        "variant": "keep_square_b128",
        "script": "bench_rmsnorm_fp32_1024x1024.py",
    },
}

CONFIG_ORDER = [
    f"{dtype}_{size}x{size}"
    for dtype in ("bf16", "fp16", "fp32")
    for size in (64, 128, 256, 512, 1024)
]


def parse_avg_time(stdout):
    matches = re.findall(r"Average execution time:\s+([\d.]+)\s+ms\s+\((\d+)\s+us\)", stdout)
    if matches:
        ms, us = matches[-1]
        return float(ms), int(us)
    match = re.search(r"Average execution time:\s+([\d.]+)\s+ms", stdout)
    if match:
        ms = float(match.group(1))
        return ms, int(round(ms * 1000))
    return None, None


def parse_backend_results(text):
    results = {}
    pattern = r"\[(tilelang-opt|PPL)\]\s+correct=(True|False)\s+max_diff=([\d.eE+-]+)\s+avg_diff=([\d.eE+-]+)"
    for match in re.finditer(pattern, text):
        backend, correct, max_diff, avg_diff = match.groups()
        key = "tl" if backend == "tilelang-opt" else "ppl"
        results[key] = {
            "correct": correct == "True",
            "max_diff": float(max_diff),
            "avg_diff": float(avg_diff),
        }
    return results


def run_one(config_name, out_dir, timeout):
    script = os.path.join(THIS_DIR, CONFIGS[config_name]["script"])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    stem = config_name
    stdout_path = os.path.join(out_dir, stem + ".stdout.txt")
    stderr_path = os.path.join(out_dir, stem + ".stderr.txt")

    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        status = "ok" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "timeout"
        returncode = None

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr)

    avg_matches = re.findall(r"Average execution time:\s+([\d.]+)\s+ms\s+\((\d+)\s+us\)", stdout)
    tl_avg_ms = tl_avg_us = ppl_avg_ms = ppl_avg_us = None
    if len(avg_matches) >= 1:
        tl_avg_ms = float(avg_matches[0][0])
        tl_avg_us = int(avg_matches[0][1])
    if len(avg_matches) >= 2:
        ppl_avg_ms = float(avg_matches[1][0])
        ppl_avg_us = int(avg_matches[1][1])
    elif len(avg_matches) == 0:
        tl_avg_ms, tl_avg_us = parse_avg_time(stdout)

    results = parse_backend_results(stdout + "\n" + stderr)
    return {
        "config": config_name,
        "variant": CONFIGS[config_name]["variant"],
        "status": status,
        "returncode": returncode,
        "tl_correct": results.get("tl", {}).get("correct"),
        "tl_max_diff": results.get("tl", {}).get("max_diff"),
        "tl_avg_diff": results.get("tl", {}).get("avg_diff"),
        "tl_avg_ms": tl_avg_ms,
        "tl_avg_us": tl_avg_us,
        "ppl_correct": results.get("ppl", {}).get("correct"),
        "ppl_max_diff": results.get("ppl", {}).get("max_diff"),
        "ppl_avg_diff": results.get("ppl", {}).get("avg_diff"),
        "ppl_avg_ms": ppl_avg_ms,
        "ppl_avg_us": ppl_avg_us,
        "stdout": os.path.relpath(stdout_path, REPO_ROOT),
        "stderr": os.path.relpath(stderr_path, REPO_ROOT),
    }


def format_value(value, suffix=""):
    if value is None:
        return "-"
    return f"{value}{suffix}"


def write_summary(rows, out_dir):
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "# RMSNorm Optimized Fixed Suite Summary",
        "",
        f"Run directory: `{os.path.relpath(out_dir, REPO_ROOT)}`",
        "",
        "Each row runs the fixed TileLang-TPU variant and the matching PPL baseline.",
        "",
        "| Config | Fixed TL variant | TL correct | TL avg us | PPL correct | PPL avg us | Speedup |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        speedup = "-"
        if (
            row.get("tl_avg_us")
            and row.get("ppl_avg_us")
            and row.get("tl_correct") is True
            and row.get("ppl_correct") is True
        ):
            speedup = f"{row['ppl_avg_us'] / row['tl_avg_us']:.2f}x"
        lines.append(
            "| {config} | {variant} | {tl_ok} | {tl_us} | {ppl_ok} | {ppl_us} | {speedup} |".format(
                config=row["config"],
                variant=row["variant"],
                tl_ok=format_value(row.get("tl_correct")),
                tl_us=format_value(row.get("tl_avg_us")),
                ppl_ok=format_value(row.get("ppl_correct")),
                ppl_us=format_value(row.get("ppl_avg_us")),
                speedup=speedup,
            )
        )

    lines.extend([
        "",
        "`Speedup = PPL(us) / TL(us)`，大于 1 表示优化后的 TileLang-TPU 更快。",
    ])

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def parse_configs(text):
    if text == "all":
        return CONFIG_ORDER
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all", help="all or comma separated config names")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out-dir")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    if args.list_configs:
        for name in CONFIG_ORDER:
            print(f"{name}: {CONFIGS[name]['variant']}  {CONFIGS[name]['script']}")
        return

    configs = parse_configs(args.configs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, "fixed_" + timestamp)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    total = len(configs)
    for index, config_name in enumerate(configs, start=1):
        print(f"[{index}/{total}] {config_name} {CONFIGS[config_name]['variant']}", flush=True)
        row = run_one(config_name, out_dir, args.timeout)
        rows.append(row)
        tl_avg = f"{row['tl_avg_us']} us" if row.get("tl_avg_us") is not None else "?"
        ppl_avg = f"{row['ppl_avg_us']} us" if row.get("ppl_avg_us") is not None else "?"
        print(
            "  status={status} tl={tl_ok}/{tl_avg} ppl={ppl_ok}/{ppl_avg}".format(
                status=row["status"],
                tl_ok=row.get("tl_correct"),
                tl_avg=tl_avg,
                ppl_ok=row.get("ppl_correct"),
                ppl_avg=ppl_avg,
            ),
            flush=True,
        )

    json_path, md_path = write_summary(rows, out_dir)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
