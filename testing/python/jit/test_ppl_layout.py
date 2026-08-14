from pathlib import Path
import json

from tilelang.jit.adapter.ppl_layout import resolve_ppl_layout


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_resolve_ppl_17_layout(tmp_path):
    arch = "tpub_7_1"
    (tmp_path / "deps/chip").mkdir(parents=True)
    (tmp_path / "deps/chip/chip_map.json").write_text(
        json.dumps({"bm1690": arch}), encoding="utf-8"
    )
    for directory in (
        f"deps/chip/{arch}/TPU1686/kernel/include",
        f"deps/chip/{arch}/lib",
        "deps/common/dev/kernel",
        "deps/common/dev/utils/include",
        "deps/common/host/include",
        "deps/runtime/tpuv7-runtime/include",
        "deps/runtime/tpuv7-runtime/lib",
    ):
        (tmp_path / directory).mkdir(parents=True)
    _touch(tmp_path / "deps/common/dev/utils/src/ppl_helper.c")
    _touch(tmp_path / f"deps/chip/{arch}/lib/libtpuv7_emulator.so")

    layout = resolve_ppl_layout(str(tmp_path))

    assert layout.release == "1.7"
    assert layout.arch == arch
    assert layout.compile_definitions == ("__tpub_7_1__", "__sg2260__")
    assert layout.firmware_archive == (
        tmp_path / f"deps/chip/{arch}/lib/libfirmware_core.a"
    )


def test_resolve_legacy_layout(tmp_path):
    chip_root = tmp_path / "runtime/bm1690"
    emulator_root = chip_root / "tpuv7-runtime-emulator"
    for directory in (
        chip_root / "TPU1686/kernel/include",
        chip_root / "lib",
        tmp_path / "runtime/kernel",
        tmp_path / "runtime/customize/include",
        emulator_root / "include",
        emulator_root / "lib",
    ):
        directory.mkdir(parents=True)
    _touch(tmp_path / "runtime/customize/src/ppl_helper.c")
    _touch(emulator_root / "lib/libtpuv7_emulator.so")

    layout = resolve_ppl_layout(str(tmp_path))

    assert layout.release == "legacy"
    assert layout.arch == "bm1690"
    assert layout.compile_definitions == ("__bm1690__",)
