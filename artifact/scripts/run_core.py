#!/usr/bin/env python3
"""Run paper benchmark configurations and collect artifact results."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifact_utils import (
    ARTIFACT_ROOT,
    collect_environment,
    expand_matrix,
    load_json,
    now_stamp,
    run_entry,
    write_json,
    write_manifest,
    write_raw_jsonl,
    write_summary_files,
)


FIGURE_CONFIGS = {
    "fig4": ARTIFACT_ROOT / "configs" / "fig4_standalone_ops.json",
    "fig5": ARTIFACT_ROOT / "configs" / "fig5_fused_ops.json",
}


def load_entries(args) -> list:
    if args.config:
        return expand_matrix(load_json(Path(args.config)))
    figures = ["fig4", "fig5"] if args.figure == "all" else [args.figure]
    entries = []
    for figure in figures:
        entries.extend(expand_matrix(load_json(FIGURE_CONFIGS[figure])))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("fig4", "fig5", "all"), default="all")
    parser.add_argument("--config")
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stamp = now_stamp()
    label = args.figure if not args.config else Path(args.config).stem
    out_dir = Path(args.out_dir) if args.out_dir else ARTIFACT_ROOT / "results" / f"{label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(args)
    if args.limit is not None:
        entries = entries[:args.limit]

    write_json(out_dir / "environment.json", collect_environment())
    rows = []
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        print(f"[{index}/{total}] {entry['name']}")
        row = run_entry(entry, out_dir, dry_run=args.dry_run)
        rows.append(row)
        if args.dry_run:
            print("  dry-run:", " ".join(row["command"]))
        else:
            print(f"  status={row['status']} timings={len(row.get('timings', []))}")

    raw_path = out_dir / "raw_results.jsonl"
    write_raw_jsonl(raw_path, rows)
    extra = write_summary_files(out_dir, rows)
    manifest_path = write_manifest(out_dir, rows, extra=extra)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

