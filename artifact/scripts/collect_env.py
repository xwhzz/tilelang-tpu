#!/usr/bin/env python3
"""Collect artifact environment metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifact_utils import collect_environment, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(Path(args.out), collect_environment())
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

