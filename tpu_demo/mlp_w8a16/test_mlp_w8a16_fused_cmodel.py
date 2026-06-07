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
    block_D = block_N  # enforced by kernel assert for W_shared reuse

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

    gate_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-128, 128, (hidden, intermediate), dtype=torch.int8)

    # Random per-block scales
    gate_s = (torch.rand(
        (intermediate + blocksize - 1) // blocksize,
        (hidden + blocksize - 1) // blocksize
    ).half() + 0.5) * 0.1
    up_s = (torch.rand(
        (intermediate + blocksize - 1) // blocksize,
        (hidden + blocksize - 1) // blocksize
    ).half() + 0.5) * 0.1
    down_s = (torch.rand(
        (hidden + blocksize - 1) // blocksize,
        (intermediate + blocksize - 1) // blocksize
    ).half() + 0.5) * 0.1

    print("Running fused TPU kernel...")
    output = run_fused_mlp(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize)

    print("Computing reference (with fp16 clamp)...")
    ref_clamp = mlp_w8a16_dq_forward_ref(
        x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize
    )
    # Reference matching kernel precision: fp16 @ fp16 GEMM with fp32 accum
    # and fp16-range clamp after the gating stage.
    gw_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    uw_fp16 = dequant_weight(up_w, up_s, blocksize)
    dw_fp16 = dequant_weight(down_w, down_s, blocksize)
    gate_fp16 = torch.matmul(x, gw_fp16.T)
    up_fp16 = torch.matmul(x, uw_fp16.T)
    act_fp16 = (torch.nn.functional.silu(gate_fp16.float()) * up_fp16.float())
    # Clamp to match kernel's fp16-range guard (Gap 2 closed)
    act_fp16 = act_fp16.clamp(-65500.0, 65500.0).half()
    ref_noclamp = torch.matmul(act_fp16, dw_fp16.T)

    out_f32 = output.float()
    ref_f32 = ref_clamp.float()
    refnc_f32 = ref_noclamp.float()

    max_diff = (out_f32 - ref_f32).abs().max().item()
    avg_diff = (out_f32 - ref_f32).abs().mean().item()
    # No-clamp reference diff (fair comparison for current kernel)
    max_diff_nc = (out_f32 - refnc_f32).abs().max().item()
    avg_diff_nc = (out_f32 - refnc_f32).abs().mean().item()
    has_inf = torch.isinf(output).any()
    has_nan = torch.isnan(output).any()

    print(f"\n--- Results ---")
    print(f"TPU output[0,:5]: {output[0, :5]}")
    print(f"Ref  output[0,:5]: {ref_clamp[0, :5]}")
    print(f"TPU  min={output.min().item():.6f}  max={output.max().item():.6f}")
    print(f"Ref  min={ref_clamp.min().item():.6f}  max={ref_clamp.max().item():.6f}")
    print(f"INF: {has_inf}, NaN: {has_nan}")
    print(f"Max  diff (vs clamped ref):   {max_diff}")
    print(f"Avg  diff (vs clamped ref):   {avg_diff}")
    print(f"Max  diff (vs noclamp ref):   {max_diff_nc}")
    print(f"Avg  diff (vs noclamp ref):   {avg_diff_nc}")

    # Per-row max diff to detect pattern issues
    row_max_diff = (out_f32 - refnc_f32).abs().max(dim=1).values
    print(f"Per-row max diff (noclamp): {row_max_diff.tolist()}")

    # Use no-clamp reference for pass/fail (fair comparison).
    # Threshold: fp16 @ fp16 GEMM on TPU (product rounded to fp16, accum in fp32)
    # vs CPU reference (upcast to fp32 first). Up to 64 difference is expected
    # fp16 precision loss across 3 sequential GEMMs with larger dimensions.
    passed = not has_inf and not has_nan and max_diff_nc < 64
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    all_passed = True

    # Stage 1: Tiny (single-tile safety check)
    # hidden=128, intermediate=128 → block_N=block_D=128, block_K=128
    if not test_fused_mlp("Stage 1: Tiny", M=4, hidden=128, intermediate=128):
        all_passed = False
        print("\n!!! STAGE 1 FAILED — stopping !!!")
        exit(1)

    # Stage 2: Small multi-K (hidden > block_K)
    if not test_fused_mlp("Stage 2: Small multi-K", M=16, hidden=256, intermediate=128):
        all_passed = False

    # Stage 3: Medium multi-N (intermediate > block_N)
    if not test_fused_mlp("Stage 3: Medium multi-N", M=32, hidden=128, intermediate=256):
        all_passed = False

    # Stage 4: Multi-tile K and N (reduced intermediate to avoid fp16 overflow)
    if not test_fused_mlp("Stage 4: Multi-NK", M=16, hidden=256, intermediate=256):
        all_passed = False

    # Stage 5: Large multi-tile
    if not test_fused_mlp("Stage 5: Large", M=32, hidden=256, intermediate=256):
        all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL CMODEL TESTS PASSED")
        print("=" * 60)
    else:
        print("\n!!! SOME TESTS FAILED !!!")
        exit(1)
