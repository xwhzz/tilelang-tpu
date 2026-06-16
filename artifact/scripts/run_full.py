#!/usr/bin/env python3
"""Run the full configured artifact suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from artifact_utils import ARTIFACT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ARTIFACT_ROOT / "scripts" / "run_core.py"),
        "--figure",
        "all",
    ]
    if args.out_dir:
        command.extend(["--out-dir", args.out_dir])
    if args.dry_run:
        command.append("--dry-run")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    print(" ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
