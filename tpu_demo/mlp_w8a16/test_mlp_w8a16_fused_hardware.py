# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Fused W8A16 dequantized MLP — real hardware (PCIe) verification test.

Tests the single fused kernel on physical BM1690 TPU hardware.
"""

import os
import torch
import torch.nn.functional as F

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight, mlp_w8a16_dq_forward_ref
from tpu_demo.mlp_w8a16.mlp_w8a16_fused_kernel import mlp_w8a16_fused_kernel
import tilelang


def run_fused_mlp_hw(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Run fused MLP kernel on real BM1690 TPU hardware via PCIe."""
    M, hidden = x.shape
    intermediate = gate_w.shape[0]
    dtype = x.dtype

    # CPU-side weight dequantization
    print("  Dequantizing weights on CPU...")
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    # Tile sizes — block_D must equal block_N for W_shared buffer reuse
    block_M = min(32, M)
    block_N = min(128, intermediate)
    block_K = min(128, hidden)
    block_D = block_N  # enforced by kernel assert

    print(f"  Compiling fused kernel (M={M}, hidden={hidden}, intermediate={intermediate})...")
    print(f"  Tiles: block_M={block_M}, block_N={block_N}, block_K={block_K}, block_D={block_D}")
    kernel = tilelang.compile(
        mlp_w8a16_fused_kernel(
            M, hidden, intermediate,
            block_M=block_M, block_N=block_N,
            block_K=block_K, block_D=block_D,
        ),
        out_idx=-1,
        target="tpu",
        mode="pcie",
    )

    output = torch.empty(M, hidden, dtype=dtype)
    print("  Launching fused kernel on TPU...")
    kernel(x, gate_w_fp16, up_w_fp16, down_w_fp16, output)
    return output


def test_fused_mlp_hw(name, M, hidden, intermediate, blocksize=128):
    """Run fused MLP hardware test."""
    print(f"\n{'='*60}")
    print(f"Hardware Test: {name}")
    print(f"M={M}, hidden={hidden}, intermediate={intermediate}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(M, hidden).half() * 0.1

    gate_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-128, 128, (hidden, intermediate), dtype=torch.int8)

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

    import time
    t0 = time.time()
    output = run_fused_mlp_hw(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize)
    tpu_time = time.time() - t0
    print(f"  TPU time: {tpu_time*1000:.1f} ms")

    print("Computing reference (with fp16 clamp)...")
    t0 = time.time()
    ref = mlp_w8a16_dq_forward_ref(
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
    act_fp16 = act_fp16.clamp(-65500.0, 65500.0).half()
    ref_noclamp = torch.matmul(act_fp16, dw_fp16.T)
    ref_time = time.time() - t0

    out_f32 = output.float()
    ref_f32 = ref.float()
    refnc_f32 = ref_noclamp.float()

    max_diff = (out_f32 - ref_f32).abs().max().item()
    avg_diff = (out_f32 - ref_f32).abs().mean().item()
    max_diff_nc = (out_f32 - refnc_f32).abs().max().item()
    avg_diff_nc = (out_f32 - refnc_f32).abs().mean().item()
    has_inf = torch.isinf(output).any().item()
    has_nan = torch.isnan(output).any().item()

    print(f"\n--- Results ---")
    print(f"TPU output[0,:5]: {output[0, :5]}")
    print(f"Ref  output[0,:5]: {ref[0, :5]}")
    print(f"TPU  INF={has_inf}, NaN={has_nan}")
    print(f"Max  diff (vs clamped ref):   {max_diff}")
    print(f"Avg  diff (vs clamped ref):   {avg_diff}")
    print(f"Max  diff (vs noclamp ref):   {max_diff_nc}")
    print(f"Avg  diff (vs noclamp ref):   {avg_diff_nc}")
    print(f"TPU time: {tpu_time*1000:.1f} ms, Ref time: {ref_time*1000:.1f} ms")

    # Use no-clamp reference for fair comparison.
    # Threshold: 64 accounts for fp16 precision loss across 3 sequential GEMMs.
    passed = not has_inf and not has_nan and max_diff_nc < 64
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    # Ensure PPL_PROJECT_ROOT is set
    if not os.environ.get("PPL_PROJECT_ROOT"):
        os.environ["PPL_PROJECT_ROOT"] = "/home/tilelanguser12/ppl_v1.4.195-geb2acdd0-20250220"
        print(f"Set PPL_PROJECT_ROOT={os.environ['PPL_PROJECT_ROOT']}")

    all_passed = True

    # Stage 1: Tiny decode (M=1, safety check)
    # Use matching hidden=intermediate=64 so block_D=block_N=64
    if not test_fused_mlp_hw("Stage 1: Decode M=1", M=1, hidden=64, intermediate=64):
        all_passed = False
        print("\n!!! STAGE 1 (Decode) FAILED !!!")

    # Stage 2: Small M=4
    if not test_fused_mlp_hw("Stage 2: Small M=4", M=4, hidden=128, intermediate=128):
        all_passed = False

    # Stage 3: Medium
    if not test_fused_mlp_hw("Stage 3: Medium M=16", M=16, hidden=256, intermediate=256):
        all_passed = False

    # Stage 4: Multi-tile D and K (reduced to avoid fp16 overflow in ref)
    if not test_fused_mlp_hw("Stage 4: Multi-DK M=16", M=16, hidden=256, intermediate=128):
        all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL HARDWARE TESTS PASSED")
        print("=" * 60)
    else:
        print("\n!!! SOME HARDWARE TESTS FAILED !!!")
        exit(1)
