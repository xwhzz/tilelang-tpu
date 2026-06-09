"""Run fixed optimized FlashAttention benchmark scripts."""

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
RESULTS_DIR = os.path.join(BENCHMARK_ROOT, "results", "flash_attention_opt")

CONFIGS = {
    "bf16_b1_h8_s64_d32": {
        "tile": "64x64",
        "script": "bench_flashattn_bf16_b1_h8_s64_d32.py",
    },
    "bf16_b1_h8_s128_d64": {
        "tile": "128x128",
        "script": "bench_flashattn_bf16_b1_h8_s128_d64.py",
    },
    "bf16_b1_h8_s256_d128": {
        "tile": "128x128",
        "script": "bench_flashattn_bf16_b1_h8_s256_d128.py",
    },
    "bf16_b1_h8_s512_d128": {
        "tile": "128x128",
        "script": "bench_flashattn_bf16_b1_h8_s512_d128.py",
    },
    "bf16_b8_h8_s512_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_bf16_b8_h8_s512_d128.py",
    },
    "bf16_b8_h32_s512_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_bf16_b8_h32_s512_d128.py",
    },
    "bf16_b64_h64_s1024_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_bf16_b64_h64_s1024_d128.py",
    },
    "bf16_b64_h64_s2048_d128": {
        "tile": "128x512_h8_s2",
        "script": "bench_flashattn_bf16_b64_h64_s2048_d128.py",
    },
    "fp16_b1_h8_s64_d32": {
        "tile": "64x64",
        "script": "bench_flashattn_fp16_b1_h8_s64_d32.py",
    },
    "fp16_b1_h8_s128_d64": {
        "tile": "128x128",
        "script": "bench_flashattn_fp16_b1_h8_s128_d64.py",
    },
    "fp16_b1_h8_s256_d128": {
        "tile": "128x128",
        "script": "bench_flashattn_fp16_b1_h8_s256_d128.py",
    },
    "fp16_b1_h8_s512_d128": {
        "tile": "128x128",
        "script": "bench_flashattn_fp16_b1_h8_s512_d128.py",
    },
    "fp16_b8_h8_s512_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_fp16_b8_h8_s512_d128.py",
    },
    "fp16_b8_h32_s512_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_fp16_b8_h32_s512_d128.py",
    },
    "fp16_b64_h64_s1024_d128": {
        "tile": "256x256_h8_s2",
        "script": "bench_flashattn_fp16_b64_h64_s1024_d128.py",
    },
    "fp16_b64_h64_s2048_d128": {
        "tile": "128x512_h8_s2",
        "script": "bench_flashattn_fp16_b64_h64_s2048_d128.py",
    },
}

CONFIG_ORDER = [
    "bf16_b1_h8_s64_d32",
    "bf16_b1_h8_s128_d64",
    "bf16_b1_h8_s256_d128",
    "bf16_b1_h8_s512_d128",
    "bf16_b8_h8_s512_d128",
    "bf16_b8_h32_s512_d128",
    "bf16_b64_h64_s1024_d128",
    "bf16_b64_h64_s2048_d128",
    "fp16_b1_h8_s64_d32",
    "fp16_b1_h8_s128_d64",
    "fp16_b1_h8_s256_d128",
    "fp16_b1_h8_s512_d128",
    "fp16_b8_h8_s512_d128",
    "fp16_b8_h32_s512_d128",
    "fp16_b64_h64_s1024_d128",
    "fp16_b64_h64_s2048_d128",
]


def parse_backend_results(text):
    results = {}
    pattern = (
        r"\[(tilelang-opt|PPL)\]\s+"
        r"(?:correct|sampled_correct|full_correct)=(True|False)\s+"
        r"max_diff=([\d.eE+-]+)\s+avg_diff=([\d.eE+-]+)"
    )
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
    stem = config_name
    stdout_path = os.path.join(out_dir, stem + ".stdout.txt")
    stderr_path = os.path.join(out_dir, stem + ".stderr.txt")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

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
        returncode = proc.returncode
        status = "ok" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = None
        status = "timeout"

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(stderr)

    matches = re.findall(r"Average execution time:\s+([\d.]+)\s+ms\s+\((\d+)\s+us\)", stdout)
    tl_avg_ms = tl_avg_us = ppl_avg_ms = ppl_avg_us = None
    if len(matches) >= 1:
        tl_avg_ms = float(matches[0][0])
        tl_avg_us = int(matches[0][1])
    if len(matches) >= 2:
        ppl_avg_ms = float(matches[1][0])
        ppl_avg_us = int(matches[1][1])

    results = parse_backend_results(stdout + "\n" + stderr)
    return {
        "config": config_name,
        "tile": CONFIGS[config_name]["tile"],
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


def format_value(value):
    if value is None:
        return "-"
    return str(value)


def write_summary(rows, out_dir):
    json_path = os.path.join(out_dir, "summary.json")
    md_path = os.path.join(out_dir, "summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "# FlashAttention Optimized Fixed Suite Summary",
        "",
        f"Run directory: `{os.path.relpath(out_dir, REPO_ROOT)}`",
        "",
        "Each row runs the fixed TileLang-TPU script and the matching PPL baseline.",
        "",
        "| Config | Fixed TL tile | TL correct | TL avg us | PPL correct | PPL avg us | Speedup |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for row in rows:
        speedup = "-"
        if row.get("tl_avg_us") and row.get("ppl_avg_us"):
            speedup = f"{row['ppl_avg_us'] / row['tl_avg_us']:.2f}x"
        lines.append(
            "| {config} | {tile} | {tl_ok} | {tl_us} | {ppl_ok} | {ppl_us} | {speedup} |".format(
                config=row["config"],
                tile=row["tile"],
                tl_ok=format_value(row.get("tl_correct")),
                tl_us=format_value(row.get("tl_avg_us")),
                ppl_ok=format_value(row.get("ppl_correct")),
                ppl_us=format_value(row.get("ppl_avg_us")),
                speedup=speedup,
            )
        )
    lines.append("")
    lines.append("`Speedup = PPL(us) / TL(us)`，大于 1 表示优化后的 TileLang-TPU 更快。")

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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out-dir")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    if args.list_configs:
        for name in CONFIG_ORDER:
            print(f"{name}: {CONFIGS[name]['tile']}  {CONFIGS[name]['script']}")
        return

    configs = parse_configs(args.configs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, "fixed_" + timestamp)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    total = len(configs)
    for index, config_name in enumerate(configs, start=1):
        print(f"[{index}/{total}] {config_name} {CONFIGS[config_name]['tile']}", flush=True)
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
