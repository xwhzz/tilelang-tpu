#!/usr/bin/env python3
"""Run representative pipeline/address-allocation ablations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from artifact_utils import (
    ARTIFACT_ROOT,
    REPO_ROOT,
    collect_environment,
    load_json,
    now_stamp,
    run_entry,
    write_json,
    write_manifest,
    write_raw_jsonl,
    write_summary_files,
)


def make_temp_stage0(case):
    source = REPO_ROOT / case["script"]
    text = source.read_text(encoding="utf-8")
    if case["stage_pattern"] not in text:
        raise RuntimeError(f"Pattern not found in {source}: {case['stage_pattern']}")
    temp = source.with_name("_artifact_tmp_" + source.name)
    temp.write_text(text.replace(case["stage_pattern"], case["stage_replacement"], 1),
                    encoding="utf-8")
    return temp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ARTIFACT_ROOT / "configs" / "fig6_ablation.json"))
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    stamp = now_stamp()
    out_dir = Path(args.out_dir) if args.out_dir else ARTIFACT_ROOT / "results" / f"fig6_ablation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "environment.json", collect_environment())

    rows = []
    total = len(config["cases"]) * len(config["variants"])
    index = 0
    temp_files = []
    try:
        for case in config["cases"]:
            for variant in config["variants"]:
                index += 1
                script = case["script"]
                if variant.get("pipeline_off"):
                    temp = make_temp_stage0(case)
                    temp_files.append(temp)
                    script = str(temp.relative_to(REPO_ROOT))
                env = {}
                if variant.get("address_off"):
                    env["TL_TPU_DISABLE_LIVE_RANGE"] = "1"
                    env["TL_TPU_DISABLE_BANK_CONFLICT_AWARE"] = "1"
                entry = {
                    "name": f"{case['name']}__{variant['name']}",
                    "figure": "fig6",
                    "operator": case["operator"],
                    "dtype": case["dtype"],
                    "shape": case["shape"],
                    "script": script,
                    "timeout_sec": case.get("timeout_sec", 900),
                    "env": env,
                }
                print(f"[{index}/{total}] {entry['name']}")
                row = run_entry(entry, out_dir, dry_run=args.dry_run)
                row["variant"] = variant
                rows.append(row)
                if args.dry_run:
                    print("  dry-run:", " ".join(row["command"]))
                else:
                    print(f"  status={row['status']} timings={len(row.get('timings', []))}")
    finally:
        for temp in temp_files:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    raw_path = out_dir / "raw_results.jsonl"
    write_raw_jsonl(raw_path, rows)
    extra = write_summary_files(out_dir, rows)
    manifest_path = write_manifest(out_dir, rows, extra=extra)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

