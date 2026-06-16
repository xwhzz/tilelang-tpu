"""Shared helpers for TileLang-TPU artifact scripts."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifact"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_value(args: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_environment() -> Dict[str, Any]:
    keys = [
        "PPL_PROJECT_ROOT",
        "TPU_KERNEL_PATH",
        "PPL_KERNEL_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "TMPDIR",
    ]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_status_short": git_value(["status", "--short"]),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "env": {key: os.environ.get(key) for key in keys},
    }


def expand_matrix(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(config.get("benchmarks", []))
    for item in config.get("matrix", []):
        for dtype in item.get("dtypes", []):
            for size in item.get("sizes", []):
                values = {"dtype": dtype, "size": size}
                script = item["script_template"].format(**values)
                shape = item.get("shape_template", "{size}").format(**values)
                rows.append({
                    "name": f"{item['operator']}_{dtype}_{shape}",
                    "figure": item.get("figure"),
                    "operator": item["operator"],
                    "dtype": dtype,
                    "shape": shape,
                    "script": script,
                    "timeout_sec": item.get("timeout_sec", 300),
                })
    return rows


def parse_correctness(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"\[(?P<backend>[^\]]+)\]\s+"
        r"(?:(?:sampled_|full_)?correct)=(?P<correct>True|False)\s+"
        r"max_diff=(?P<max>[\d.eE+-]+)\s+"
        r"avg_diff=(?P<avg>[\d.eE+-]+)"
    )
    for match in pattern.finditer(text):
        rows.append({
            "backend": match.group("backend"),
            "correct": match.group("correct") == "True",
            "max_diff": float(match.group("max")),
            "avg_diff": float(match.group("avg")),
        })

    for line in text.splitlines():
        if line.startswith("RESULT_JSON:"):
            payload = line[len("RESULT_JSON:"):]
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            rows.append({
                "backend": parsed.get("name", "RESULT_JSON"),
                "correct": parsed.get("correct"),
                "max_diff": parsed.get("max_diff"),
                "avg_diff": parsed.get("avg_diff"),
                "check": parsed.get("check"),
            })

    if "check close:" in text:
        tail = text.split("check close:", 1)[1]
        if re.search(r"\bTrue\b", tail):
            rows.append({"backend": "tilelang-cmodel", "correct": True})
        elif re.search(r"\bFalse\b", tail):
            rows.append({"backend": "tilelang-cmodel", "correct": False})
    return rows


def parse_timings(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(r"Average execution time:\s+([\d.]+)\s+ms(?:\s+\((\d+)\s+us\))?")
    for match in pattern.finditer(text):
        ms = float(match.group(1))
        us = int(match.group(2)) if match.group(2) else int(round(ms * 1000))
        rows.append({"ms": ms, "us": us})
    return rows


def parse_output(text: str) -> Dict[str, Any]:
    return {
        "timings": parse_timings(text),
        "correctness": parse_correctness(text),
        "has_traceback": "Traceback (most recent call last)" in text,
    }


def command_for_entry(entry: Dict[str, Any], out_dir: Path) -> List[str]:
    if entry.get("kind") == "command":
        args = []
        for arg in entry["args"]:
            value = arg.format(out_dir=str(out_dir))
            args.append(sys.executable if value == "python" else value)
        return args
    return [sys.executable, entry["script"]]


def run_entry(entry: Dict[str, Any], out_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    name = entry["name"]
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.txt"
    stderr_path = logs_dir / f"{name}.stderr.txt"
    command = command_for_entry(entry, out_dir)
    timeout = int(entry.get("timeout_sec", 300))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(entry.get("env", {}))

    base_row: Dict[str, Any] = {
        "name": name,
        "figure": entry.get("figure"),
        "operator": entry.get("operator"),
        "dtype": entry.get("dtype"),
        "shape": entry.get("shape"),
        "command": command,
        "timeout_sec": timeout,
        "stdout": rel(stdout_path),
        "stderr": rel(stderr_path),
    }
    if dry_run:
        write_text(stdout_path, "DRY_RUN " + " ".join(command) + "\n")
        write_text(stderr_path, "")
        return {**base_row, "status": "dry_run", "returncode": None, "timings": [], "correctness": []}

    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
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

    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    parsed = parse_output(stdout + "\n" + stderr)
    return {
        **base_row,
        "status": status,
        "returncode": returncode,
        **parsed,
    }


def write_raw_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows = []
    for row in rows:
        timings = row.get("timings") or []
        correctness = row.get("correctness") or []
        tl_us = timings[0]["us"] if len(timings) >= 1 else None
        other_us = timings[1]["us"] if len(timings) >= 2 else None
        speedup = (other_us / tl_us) if tl_us and other_us else None
        correct_values = [item.get("correct") for item in correctness if item.get("correct") is not None]
        all_correct = all(correct_values) if correct_values else None
        summary_rows.append({
            "name": row.get("name"),
            "figure": row.get("figure"),
            "operator": row.get("operator"),
            "dtype": row.get("dtype"),
            "shape": row.get("shape"),
            "status": row.get("status"),
            "all_correct": all_correct,
            "tl_us": tl_us,
            "other_us": other_us,
            "speedup": speedup,
            "stdout": row.get("stdout"),
            "stderr": row.get("stderr"),
        })
    return {
        "total": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({row.get("status") for row in rows})
        },
        "rows": summary_rows,
    }


def write_summary_files(out_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    summary = summarize_rows(rows)
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    figure_csv = out_dir / "figure_inputs.csv"
    write_json(summary_json, summary)

    lines = [
        "# Artifact Run Summary",
        "",
        f"Run directory: `{rel(out_dir)}`",
        "",
        "| Name | Status | Correct | TL us | Other us | Speedup |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        speedup = "-" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        lines.append(
            "| {name} | {status} | {correct} | {tl} | {other} | {speedup} |".format(
                name=row["name"],
                status=row["status"],
                correct=row["all_correct"],
                tl="-" if row["tl_us"] is None else row["tl_us"],
                other="-" if row["other_us"] is None else row["other_us"],
                speedup=speedup,
            )
        )
    write_text(summary_md, "\n".join(lines) + "\n")

    with figure_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "figure",
                "operator",
                "dtype",
                "shape",
                "status",
                "all_correct",
                "tl_us",
                "other_us",
                "speedup",
            ],
        )
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    return {
        "summary": rel(summary_json),
        "tables": rel(summary_md),
        "figure_inputs": rel(figure_csv),
    }


def write_manifest(out_dir: Path, rows: List[Dict[str, Any]], extra: Optional[Dict[str, str]] = None) -> Path:
    manifest = {
        "environment": rel(out_dir / "environment.json"),
        "raw_results": rel(out_dir / "raw_results.jsonl"),
        "summary": rel(out_dir / "summary.json"),
        "tables": rel(out_dir / "summary.md"),
        "figure_inputs": rel(out_dir / "figure_inputs.csv"),
        "logs": [],
    }
    for row in rows:
        if row.get("stdout"):
            manifest["logs"].append(row["stdout"])
        if row.get("stderr"):
            manifest["logs"].append(row["stderr"])
    if extra:
        manifest.update(extra)
    path = out_dir / "artifact_manifest.json"
    write_json(path, manifest)
    return path
