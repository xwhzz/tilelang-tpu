# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Target configuration shared by the TileLang TPU compilation pipeline."""

from dataclasses import dataclass
from typing import Literal, Optional


TPUDeviceMode = Literal["atomic", "rv"]
TPURuntimeMode = Literal["pcie", "cmodel"]


@dataclass(frozen=True)
class TPUCompileConfig:
    """Orthogonal TPU chip, device-code, and runtime selections."""

    chip: str = "bm1690"
    device_mode: TPUDeviceMode = "atomic"
    runtime_mode: TPURuntimeMode = "pcie"

    def __post_init__(self):
        chip = self.chip.strip().lower()
        if not chip:
            raise ValueError("TPU chip must not be empty")
        object.__setattr__(self, "chip", chip)
        if self.device_mode not in ("atomic", "rv"):
            raise ValueError(
                f"Unsupported TPU device mode {self.device_mode!r}; expected 'atomic' or 'rv'")
        if self.runtime_mode not in ("pcie", "cmodel"):
            raise ValueError(
                f"Unsupported TPU runtime mode {self.runtime_mode!r}; expected 'pcie' or 'cmodel'")


def resolve_tpu_compile_config(
    *,
    chip: str = "bm1690",
    device_mode: TPUDeviceMode = "atomic",
    runtime_mode: Optional[TPURuntimeMode] = None,
    mode: Optional[TPURuntimeMode] = None,
) -> TPUCompileConfig:
    """Normalize the new runtime_mode API and the legacy mode alias."""
    if runtime_mode is not None and mode is not None and runtime_mode != mode:
        raise ValueError(
            f"Conflicting TPU runtime modes: runtime_mode={runtime_mode!r}, mode={mode!r}")
    return TPUCompileConfig(
        chip=chip,
        device_mode=device_mode,
        runtime_mode=runtime_mode or mode or "pcie",
    )
