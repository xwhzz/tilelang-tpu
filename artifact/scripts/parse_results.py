#!/usr/bin/env python3
"""Parse artifact raw results or an existing benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_utils import load_jsonl, write_json, write_summary_files


def rows_from_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("rows", []):
        rows.append({
            "name": Path(item.get("script", f"row_{item.get('index', 0)}")).stem,
            "figure": None,
            "operator": None,
            "dtype": None,
            "shape": None,
            "status": item.get("status"),
            "returncode": item.get("returncode"),
            "stdout": item.get("stdout"),
            "stderr": item.get("stderr"),
            "timings": item.get("timings", []),
            "correctness": item.get("correctness", []),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".jsonl":
        rows = load_jsonl(input_path)
    else:
        rows = rows_from_manifest(input_path)

    write_json(out_dir / "parsed_rows.json", rows)
    outputs = write_summary_files(out_dir, rows)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

