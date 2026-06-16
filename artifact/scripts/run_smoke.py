#!/usr/bin/env python3
"""Run the TileLang-TPU artifact smoke suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from artifact_utils import ARTIFACT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pcie", "cmodel"), default="pcie")
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ARTIFACT_ROOT / "configs" / ("smoke_cmodel.json" if args.mode == "cmodel" else "smoke.json")
    command = [
        sys.executable,
        str(ARTIFACT_ROOT / "scripts" / "run_core.py"),
        "--config",
        str(config),
    ]
    if args.out_dir:
        command.extend(["--out-dir", args.out_dir])
    if args.dry_run:
        command.append("--dry-run")
    print(" ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

