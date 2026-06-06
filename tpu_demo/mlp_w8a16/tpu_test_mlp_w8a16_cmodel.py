# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Full W8A16 dequantized MLP — cmodel end-to-end test using direct kernels.

Uses direct fp16 matmul kernels (no wrapper caching) to verify correctness.
"""

import torch
import torch.nn.functional as F

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight
from tpu_demo.mlp_w8a16.tpu_mlp_w8a16_kernel import fp16_matmul_kernel
import tilelang


def run_mlp_w8a16(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Run MLP with direct kernel calls (no wrapper caching)."""
    M, hidden = x.shape
    intermediate = gate_w.shape[0]
    dtype = x.dtype

    # CPU-side weight dequantization
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    # Tile sizes
    block_M = min(32, M)
    block_N = min(128, intermediate)
    block_K = min(128, hidden)

    # Kernel 1: gate = x @ gate_w_fp16^T
    gk = tilelang.compile(
        fp16_matmul_kernel(M, intermediate, hidden, block_M, block_N, block_K),
        out_idx=-1, target="tpu", mode="cmodel",
    )
    gate_out = torch.empty(M, intermediate, dtype=dtype)
    gk(x, gate_w_fp16, gate_out)

    # Kernel 2: up = x @ up_w_fp16^T
    uk = tilelang.compile(
        fp16_matmul_kernel(M, intermediate, hidden, block_M, block_N, block_K),
        out_idx=-1, target="tpu", mode="cmodel",
    )
    up_out = torch.empty(M, intermediate, dtype=dtype)
    uk(x, up_w_fp16, up_out)

    # SiLU activation (fp32 for precision, clamp to normal fp16 range)
    act = F.silu(gate_out.float()) * up_out.float()
    act = act.clamp(-65500.0, 65500.0)
    act = torch.where(act.abs() < 6.0e-5, torch.zeros_like(act), act)
    act_h = act.to(dtype)

    # Kernel 3: output = act @ down_w_fp16^T
    block_N_down = min(128, hidden)
    block_K_down = min(128, intermediate)
    dk = tilelang.compile(
        fp16_matmul_kernel(M, hidden, intermediate, block_M, block_N_down, block_K_down),
        out_idx=-1, target="tpu", mode="cmodel",
    )
    output = torch.empty(M, hidden, dtype=dtype)
    dk(act_h, down_w_fp16, output)

    return output


def ref_mlp(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Reference MLP (matching the above, with fp16 clamp)."""
    M, hidden = x.shape
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    gate = (x.float() @ gate_w_fp16.float().T).half()
    up = (x.float() @ up_w_fp16.float().T).half()

    act = F.silu(gate.float()) * up.float()
    act = act.clamp(-65500.0, 65500.0)
    act = torch.where(act.abs() < 6.0e-5, torch.zeros_like(act), act)
    act_h = act.half()

    out = (act_h.float() @ down_w_fp16.float().T).half()
    return out


def test_mlp(M, hidden, intermediate, blocksize=128):
    """Run end-to-end MLP test."""
    print(f"\n=== MLP W8A16 Test ===")
    print(f"M={M}, hidden={hidden}, intermediate={intermediate}, blocksize={blocksize}")

    torch.manual_seed(42)
    x = torch.randn(M, hidden).half() * 0.05

    gate_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-2, 1, (hidden, intermediate), dtype=torch.int8)

    gate_s = torch.ones((intermediate + 127) // 128, (hidden + 127) // 128).half() * 0.2
    down_s = torch.ones((hidden + 127) // 128, (intermediate + 127) // 128).half() * 0.2

    print("Running TileLang-TPU MLP...")
    output = run_mlp_w8a16(x, gate_w, up_w, down_w, gate_s, gate_s, down_s, blocksize)

    print("Computing reference...")
    ref = ref_mlp(x, gate_w, up_w, down_w, gate_s, gate_s, down_s, blocksize)

    # Compare in fp32
    out_f32 = output.float()
    ref_f32 = ref.float()

    max_diff = (out_f32 - ref_f32).abs().max().item()
    avg_diff = (out_f32 - ref_f32).abs().mean().item()

    has_inf = torch.isinf(output).any() or torch.isinf(ref).any()
    has_nan = torch.isnan(output).any() or torch.isnan(ref).any()

    print(f"\n=== Results ===")
    print(f"TPU output[0,:5]: {output[0, :5]}")
    print(f"Ref  output[0,:5]: {ref[0, :5]}")
    print(f"Output INF: {torch.isinf(output).sum().item()}, NaN: {torch.isnan(output).sum().item()}")
    print(f"Ref    INF: {torch.isinf(ref).sum().item()}, NaN: {torch.isnan(ref).sum().item()}")
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {avg_diff}")

    passed = not has_inf and not has_nan and max_diff < 1.0
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    if not test_mlp(M=16, hidden=64, intermediate=128):
        print("\n!!! TEST 1 FAILED !!!")
        exit(1)
    if not test_mlp(M=32, hidden=128, intermediate=256):
        print("\n!!! TEST 2 FAILED !!!")
        exit(1)
    print("\n=== ALL MLP W8A16 TESTS PASSED ===")
