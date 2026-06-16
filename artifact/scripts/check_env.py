#!/usr/bin/env python3
"""Check the TileLang-TPU artifact environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from artifact_utils import REPO_ROOT, collect_environment, write_json


def check_import(name: str):
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pcie", "cmodel"), default="pcie")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    add("torch import", check_import("torch"))
    add("numpy import", check_import("numpy"))
    add("tilelang import", check_import("tilelang"))
    add("patches/tvm.patch", (REPO_ROOT / "patches" / "tvm.patch").exists())
    add("3rdparty/tvm exists", (REPO_ROOT / "3rdparty" / "tvm").exists())
    add("install_tpu.sh", (REPO_ROOT / "install_tpu.sh").exists())
    add("PPL_PROJECT_ROOT env", bool(os.environ.get("PPL_PROJECT_ROOT")), os.environ.get("PPL_PROJECT_ROOT", ""))
    add("TPU template dir", (REPO_ROOT / "src" / "tl_templates" / "tpu").exists())
    add("gcc available", bool(shutil.which("gcc")), shutil.which("gcc") or "")

    if args.mode == "pcie":
        add("/dev BM1690 node hint", any(Path("/dev").glob("sg*")), "looks for /dev/sg*")
    else:
        add("cmodel selected", True, "no physical BM1690 card required for correctness-only smoke")

    env = collect_environment()
    report = {"mode": args.mode, "checks": checks, "environment": env}
    if args.json_out:
        write_json(Path(args.json_out), report)

    width = max(len(item["name"]) for item in checks)
    failed = 0
    for item in checks:
        status = "OK" if item["ok"] else "WARN"
        if not item["ok"]:
            failed += 1
        print(f"{status:4} {item['name']:<{width}} {item['detail']}")

    print(f"\nGit commit: {env.get('git_commit')}")
    print(f"Mode: {args.mode}")
    if failed:
        print(f"Warnings: {failed}. Resolve WARN items before PCIe performance reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

