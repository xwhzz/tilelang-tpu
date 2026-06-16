#!/usr/bin/env python3
"""Verify that paths referenced by an artifact manifest exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_utils import REPO_ROOT


def iter_paths(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_paths(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    for raw in sorted(set(iter_paths(data))):
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            missing.append(raw)
    if missing:
        print("Missing paths:")
        for path in missing:
            print(f"  {path}")
        return 1
    print(f"Manifest OK: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

