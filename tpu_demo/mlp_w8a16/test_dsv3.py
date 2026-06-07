# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""DeepSeek-V3 Dense MLP dimensions — cmodel + pcie verification."""

import os, sys, time
import torch
from tpu_demo.mlp_w8a16.torch_ref import dequant_weight, mlp_w8a16_dq_forward_ref
from tpu_demo.mlp_w8a16.mlp_w8a16_fused_kernel import mlp_w8a16_fused_kernel
import tilelang


def test_dsv3(name, M, hidden, intermediate, mode="cmodel", blocksize=128):
    print(f"\n{'='*60}")
    print(f"DSv3 {name}: M={M} hidden={hidden} intermediate={intermediate} mode={mode}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(M, hidden).half() * 0.02  # Smaller scale for deeper net

    gate_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-128, 128, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-128, 128, (hidden, intermediate), dtype=torch.int8)

    gate_s = (torch.rand((intermediate+127)//128, (hidden+127)//128).half() + 0.5) * 0.05
    up_s = (torch.rand((intermediate+127)//128, (hidden+127)//128).half() + 0.5) * 0.05
    down_s = (torch.rand((hidden+127)//128, (intermediate+127)//128).half() + 0.5) * 0.05

    print("Dequantizing weights...")
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    block_M = min(32, M)  # support block_M=32 after buffer optimizations
    block_N = min(128, intermediate)
    block_K = min(128, hidden)
    block_D = block_N

    n_tiles = (intermediate + block_N - 1) // block_N
    k_tiles = (hidden + block_K - 1) // block_K
    d_tiles = (hidden + block_D - 1) // block_D
    print(f"Tiles: N={n_tiles} K={k_tiles} D={d_tiles} (total ops ~{n_tiles*k_tiles*d_tiles})")

    print("Compiling kernel...")
    t0 = time.time()
    kernel = tilelang.compile(
        mlp_w8a16_fused_kernel(M, hidden, intermediate,
            block_M=block_M, block_N=block_N, block_K=block_K, block_D=block_D),
        out_idx=-1, target="tpu", mode=mode,
    )
    compile_time = time.time() - t0
    print(f"Compile time: {compile_time:.1f}s")

    output = torch.zeros(M, hidden).half()
    print("Running kernel...")
    t0 = time.time()
    kernel(x, gate_w_fp16, up_w_fp16, down_w_fp16, output)
    tpu_time = time.time() - t0
    print(f"TPU time: {tpu_time*1000:.1f} ms")

    print("Computing reference...")
    t0 = time.time()
    ref = mlp_w8a16_dq_forward_ref(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize)
    ref_time = time.time() - t0

    out_f32, ref_f32 = output.float(), ref.float()
    max_diff = (out_f32 - ref_f32).abs().max().item()
    avg_diff = (out_f32 - ref_f32).abs().mean().item()
    inf_n = torch.isinf(output).sum().item()
    nan_n = torch.isnan(output).sum().item()

    print(f"Output[0,:6]: {output[0,:6]}")
    print(f"Ref[0,:6]:    {ref[0,:6]}")
    print(f"Max diff: {max_diff:.6f}  Avg diff: {avg_diff:.6f}  INF={inf_n} NaN={nan_n}")
    passed = inf_n == 0 and nan_n == 0 and max_diff < 1.0
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    if not os.environ.get("PPL_PROJECT_ROOT"):
        os.environ["PPL_PROJECT_ROOT"] = "/home/tilelanguser12/ppl_v1.4.195-geb2acdd0-20250220"

    all_ok = True

    # Stage 1: Half DSv3 (h=2048, i=5504) cmodel — validates multi-D-tile logic
    if not test_dsv3("Half-DSv3 M=1 cmodel", M=1, hidden=2048, intermediate=5504, mode="cmodel"):
        all_ok = False

    # Stage 2: Quarter DSv3 (h=1024, i=2752) pcie — validates hardware path
    if not test_dsv3("Quarter-DSv3 M=1 pcie", M=1, hidden=1024, intermediate=2752, mode="pcie"):
        all_ok = False

    if all_ok:
        print("\nALL DSv3 TESTS PASSED")
    else:
        print("\nDSv3 TESTS FAILED")
        sys.exit(1)
