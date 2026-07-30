# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import time

import pytest
import torch

from tpu_demo.ppl.rms_norm.vllm_sophgo_rmsnorm_forward.vllm_sophgo_rmsnorm_forward import (
    build_kernel,
)


def _device_for_mode(mode):
    return os.environ.get("RMS_DEVICE", "cpu" if mode == "cmodel" else "tpu:0")


def _reference(hidden_states, weight, eps):
    ref = hidden_states.float()
    ref = ref * torch.rsqrt(torch.mean(ref * ref, dim=-1, keepdim=True) + eps)
    return (ref * weight.float()).to(hidden_states.dtype)


def _synchronize_kernel(kernel):
    ret = kernel.adapter.lib.tilelang_tpu_synchronize()
    assert ret == 0


@pytest.mark.parametrize("m,n", [(128, 128), (32, 4096)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_vllm_sophgo_rmsnorm_forward(m, n, eps):
    mode = os.environ.get("RMS_MODE", "cmodel")
    device = _device_for_mode(mode)
    if device.startswith("tpu"):
        pytest.importorskip("torch_tpu")

    kernel = build_kernel(m, n, mode=mode)
    torch.manual_seed(0)
    hidden_states = torch.randn(m, n, dtype=torch.float16, device=device)
    weight = torch.randn(n, dtype=torch.float16, device=device)
    output = torch.empty_like(hidden_states)

    ret = kernel(
        hidden_states,
        weight,
        None,
        output,
        hidden_states.dim() - 1,
        eps,
    )

    assert ret == 0
    assert torch.allclose(
        output.float(),
        _reference(hidden_states, weight, eps).float(),
        atol=5e-2,
        rtol=5e-2,
    )


def test_vllm_sophgo_rmsnorm_latency():
    mode = os.environ.get("RMS_MODE", "pcie")
    if mode != "pcie":
        pytest.skip("Latency target applies to PCIe hardware execution")

    device = _device_for_mode(mode)
    if not device.startswith("tpu"):
        pytest.skip("Latency target requires TPU device tensors")
    pytest.importorskip("torch_tpu")

    m = int(os.environ.get("RMS_M", "128"))
    n = int(os.environ.get("RMS_N", "4096"))
    eps = float(os.environ.get("RMS_EPS", "1e-6"))
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    iters = int(os.environ.get("BENCH_ITERS", "100"))
    max_latency_ms = float(os.environ.get("RMS_MAX_LATENCY_MS", "1.0"))

    kernel = build_kernel(m, n, mode=mode)
    hidden_states = torch.randn(m, n, dtype=torch.float16, device=device)
    weight = torch.randn(n, dtype=torch.float16, device=device)
    output = torch.empty_like(hidden_states)
    args = (hidden_states, weight, None, output, hidden_states.dim() - 1, eps)

    for _ in range(warmup):
        assert kernel(*args) == 0
    _synchronize_kernel(kernel)

    start = time.perf_counter()
    for _ in range(iters):
        assert kernel(*args) == 0
    _synchronize_kernel(kernel)
    avg_ms = (time.perf_counter() - start) * 1000.0 / iters

    assert avg_ms < max_latency_ms, (
        f"RMSNorm average latency {avg_ms:.3f} ms exceeds "
        f"{max_latency_ms:.3f} ms"
    )


if __name__ == "__main__":
    test_vllm_sophgo_rmsnorm_forward(
        int(os.environ.get("RMS_M", "128")),
        int(os.environ.get("RMS_N", "128")),
        float(os.environ.get("RMS_EPS", "1e-6")),
    )
