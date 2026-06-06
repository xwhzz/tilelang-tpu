# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused W8A16 dequantized MLP — cmodel verification test.

Tests the single fused kernel that performs gate + up matmuls,
SiLU, element-wise multiply, and down matmul in one TPU launch.
"""

import torch
import torch.nn.functional as F

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight, mlp_w8a16_dq_forward_ref
from tpu_demo.mlp_w8a16.mlp_w8a16_fused_kernel import mlp_w8a16_fused_kernel
import tilelang


def run_fused_mlp(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Run fused MLP kernel on cmodel."""
    M, hidden = x.shape
    intermediate = gate_w.shape[0]
    dtype = x.dtype

    # CPU-side weight dequantization (one-time cost)
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    # Tile sizes
    block_M = min(32, M)
    block_N = min(128, intermediate)
    block_K = min(128, hidden)
    block_D = min(128, hidden)

    kernel = tilelang.compile(
        mlp_w8a16_fused_kernel(
            M, hidden, intermediate,
            block_M=block_M, block_N=block_N,
            block_K=block_K, block_D=block_D,
        ),
        out_idx=-1,
        target="tpu",
        mode="cmodel",
    )

    output = torch.empty(M, hidden, dtype=dtype)
    kernel(x, gate_w_fp16, up_w_fp16, down_w_fp16, output)
    return output


def test_fused_mlp(name, M, hidden, intermediate, blocksize=128):
    """Run fused MLP cmodel test."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"M={M}, hidden={hidden}, intermediate={intermediate}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(M, hidden).half() * 0.1

    gate_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-2, 1, (hidden, intermediate), dtype=torch.int8)

    # Uniform scale for simplicity
    gate_s = torch.ones(
        (intermediate + blocksize - 1) // blocksize,
        (hidden + blocksize - 1) // blocksize
    ).half() * 0.2
    up_s = gate_s  # same scale for gate and up
    down_s = torch.ones(
        (hidden + blocksize - 1) // blocksize,
        (intermediate + blocksize - 1) // blocksize
    ).half() * 0.2

    print("Running fused TPU kernel...")
    output = run_fused_mlp(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize)

    print("Computing reference...")
    ref = mlp_w8a16_dq_forward_ref(
        x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize
    )

    out_f32 = output.float()
    ref_f32 = ref.float()

    max_diff = (out_f32 - ref_f32).abs().max().item()
    avg_diff = (out_f32 - ref_f32).abs().mean().item()
    has_inf = torch.isinf(output).any() or torch.isinf(ref).any()
    has_nan = torch.isnan(output).any() or torch.isnan(ref).any()

    print(f"\n--- Results ---")
    print(f"TPU output[0,:5]: {output[0, :5]}")
    print(f"Ref  output[0,:5]: {ref[0, :5]}")
    print(f"TPU  min={output.min().item():.6f}  max={output.max().item():.6f}")
    print(f"Ref  min={ref.min().item():.6f}  max={ref.max().item():.6f}")
    print(f"INF: {torch.isinf(output).sum().item()}, NaN: {torch.isnan(output).sum().item()}")
    print(f"Ref INF: {torch.isinf(ref).sum().item()}, NaN: {torch.isnan(ref).sum().item()}")
    print(f"Max  diff: {max_diff}")
    print(f"Avg  diff: {avg_diff}")

    # Per-row max diff to detect pattern issues
    row_max_diff = (out_f32 - ref_f32).abs().max(dim=1).values
    print(f"Per-row max diff: {row_max_diff.tolist()}")

    passed = not has_inf and not has_nan and max_diff < 1.0
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    all_passed = True

    # Stage 1: Tiny (safety check)
    if not test_fused_mlp("Stage 1: Tiny", M=4, hidden=32, intermediate=64):
        all_passed = False
        print("\n!!! STAGE 1 FAILED — stopping !!!")
        exit(1)

    # Stage 2: Small (multi-tile verification)
    if not test_fused_mlp("Stage 2: Small", M=16, hidden=64, intermediate=128):
        all_passed = False

    # Stage 3: Medium (larger tiles)
    if not test_fused_mlp("Stage 3: Medium", M=32, hidden=128, intermediate=256):
        all_passed = False

    # Stage 4: Multi-tile K dimension (hidden > 128)
    if not test_fused_mlp("Stage 4: Multi-K", M=16, hidden=256, intermediate=512):
        all_passed = False

    # Stage 5: Multi-tile N and K
    if not test_fused_mlp("Stage 5: Multi-NK", M=32, hidden=256, intermediate=512):
        all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL CMODEL TESTS PASSED")
        print("=" * 60)
    else:
        print("\n!!! SOME TESTS FAILED !!!")
        exit(1)
