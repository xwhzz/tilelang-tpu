#!/usr/bin/env python3
"""Copy submitted paper figures into a result directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from artifact_utils import ARTIFACT_ROOT, rel, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = ARTIFACT_ROOT / "figures" / "paper"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source.rglob("*"):
        if path.is_file():
            dest = out_dir / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied.append(rel(dest))
    manifest = out_dir / "figures_manifest.json"
    write_json(manifest, {"copied": copied})
    print(f"Copied {len(copied)} files to {out_dir}")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

