# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Full W8A16 dequantized MLP — hardware (PCIe) end-to-end test.

SAFETY: Start with small dimensions to verify kernel correctness
before scaling up to realistic DeepSeek-V3 sizes.
"""

import torch
import torch.nn.functional as F

from tpu_demo.mlp_w8a16.torch_ref import dequant_weight
from tpu_demo.mlp_w8a16.tpu_mlp_w8a16_kernel import fp16_matmul_kernel
import tilelang


def run_mlp_w8a16(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Run MLP with direct kernel calls on hardware (mode='pcie')."""
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
    print("  Compiling gate kernel...")
    gk = tilelang.compile(
        fp16_matmul_kernel(M, intermediate, hidden, block_M, block_N, block_K),
        out_idx=-1, target="tpu",
    )
    gate_out = torch.empty(M, intermediate, dtype=dtype)
    print("  Running gate kernel...")
    gk(x, gate_w_fp16, gate_out)

    # Kernel 2: up = x @ up_w_fp16^T
    print("  Compiling up kernel...")
    uk = tilelang.compile(
        fp16_matmul_kernel(M, intermediate, hidden, block_M, block_N, block_K),
        out_idx=-1, target="tpu",
    )
    up_out = torch.empty(M, intermediate, dtype=dtype)
    print("  Running up kernel...")
    uk(x, up_w_fp16, up_out)

    # SiLU activation — keep in fp32 for precision
    act = F.silu(gate_out.float()) * up_out.float()
    act = act.clamp(-65500.0, 65500.0)
    act = act.half()

    # Kernel 3: output = act @ down_w_fp16^T
    block_N_down = min(128, hidden)
    block_K_down = min(128, intermediate)
    print("  Compiling down kernel...")
    dk = tilelang.compile(
        fp16_matmul_kernel(M, hidden, intermediate, block_M, block_N_down, block_K_down),
        out_idx=-1, target="tpu",
    )
    output = torch.empty(M, hidden, dtype=dtype)
    print("  Running down kernel...")
    dk(act, down_w_fp16, output)

    return output


def ref_mlp(x, gate_w, up_w, down_w, gate_s, up_s, down_s, blocksize=128):
    """Reference MLP in fp32 for accuracy comparison."""
    gate_w_fp16 = dequant_weight(gate_w, gate_s, blocksize)
    up_w_fp16 = dequant_weight(up_w, up_s, blocksize)
    down_w_fp16 = dequant_weight(down_w, down_s, blocksize)

    gate = (x.float() @ gate_w_fp16.float().T).half()
    up = (x.float() @ up_w_fp16.float().T).half()

    act = F.silu(gate.float()) * up.float()
    act = act.clamp(-65500.0, 65500.0)
    act_h = act.half()

    out = (act_h.float() @ down_w_fp16.float().T).half()
    return out


def test_mlp(name, M, hidden, intermediate, blocksize=128):
    """Run end-to-end MLP test on hardware."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"M={M}, hidden={hidden}, intermediate={intermediate}, blocksize={blocksize}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(M, hidden).half() * 0.1

    gate_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    up_w = torch.randint(-2, 1, (intermediate, hidden), dtype=torch.int8)
    down_w = torch.randint(-2, 1, (hidden, intermediate), dtype=torch.int8)

    gate_s = torch.ones((intermediate + 127) // 128, (hidden + 127) // 128).half() * 0.2
    down_s = torch.ones((hidden + 127) // 128, (intermediate + 127) // 128).half() * 0.2

    print("Running TileLang-TPU MLP...")
    output = run_mlp_w8a16(x, gate_w, up_w, down_w, gate_s, gate_s, down_s, blocksize)

    print("Computing reference...")
    ref = ref_mlp(x, gate_w, up_w, down_w, gate_s, gate_s, down_s, blocksize)

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
    print(f"Output INF: {torch.isinf(output).sum().item()}, NaN: {torch.isnan(output).sum().item()}")
    print(f"Ref    INF: {torch.isinf(ref).sum().item()}, NaN: {torch.isnan(ref).sum().item()}")
    print(f"Max  diff: {max_diff}")
    print(f"Avg  diff: {avg_diff}")

    # Also check per-row max diff to detect pattern issues
    row_max_diff = (out_f32 - ref_f32).abs().max(dim=1).values
    print(f"Per-row max diff: {row_max_diff.tolist()}")

    passed = not has_inf and not has_nan and max_diff < 1.0
    print(f"PASS: {passed}")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("W8A16 Dequant MLP — Hardware (PCIe) Test Suite")
    print("=" * 60)

    all_passed = True

    # Stage 1: Tiny test (safety check)
    if not test_mlp("Stage 1: Tiny", M=4, hidden=32, intermediate=64):
        print("\n!!! STAGE 1 FAILED — stopping for safety !!!")
        all_passed = False
        # Continue anyway to gather more diagnostics

    # Stage 2: Small test
    if not test_mlp("Stage 2: Small", M=16, hidden=64, intermediate=128):
        print("\n!!! STAGE 2 FAILED !!!")
        all_passed = False

    # Stage 3: Medium test
    if not test_mlp("Stage 3: Medium", M=32, hidden=256, intermediate=512):
        print("\n!!! STAGE 3 FAILED !!!")
        all_passed = False

    # Stage 4: Realistic test (DeepSeek-V3 dense MLP dimensions)
    if not test_mlp("Stage 4: DeepSeek-Dense", M=64, hidden=4096, intermediate=11008):
        print("\n!!! STAGE 4 FAILED !!!")
        all_passed = False

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL HARDWARE TESTS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED — see diagnostics above")
        print("=" * 60)
