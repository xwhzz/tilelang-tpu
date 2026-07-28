# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Test llama_mlp_gptq_forward on TPU (cmodel or hardware)."""

import argparse
import torch
import tilelang
from examples.llama_mlp_gptq.llama_mlp_gptq import llama_mlp_gptq, reference_llama_mlp_gptq


def make_fake_gptq_weight(rows, cols, group_size, seed):
    """Generate fake GPTQ weights with packed uint4 weights/zp and per-group scale."""
    torch.manual_seed(seed)
    num_elems = 2
    groups = cols // group_size
    packed_cols = cols // num_elems

    w_int4 = torch.randint(0, 16, (rows, cols))
    w_packed = torch.zeros((rows, packed_cols), dtype=torch.uint8)
    for j in range(cols):
        shift = 4 * (j % num_elems)
        w_packed[:, j // num_elems] |= (w_int4[:, j] & 0xF).to(torch.uint8) << shift

    zp_packed = torch.zeros((rows, (groups + num_elems - 1) // num_elems), dtype=torch.uint8)
    scale = torch.ones(rows, groups, dtype=torch.float16) * 0.01
    return w_packed, zp_packed, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Run on real TPU hardware")
    parser.add_argument("-M", type=int, default=128, help="Matrix M dimension")
    parser.add_argument("-K", type=int, default=128, help="Matrix K dimension")
    parser.add_argument("-B", "--block", type=int, default=128, help="Tile block size")
    parser.add_argument("-G", "--group_size", type=int, default=128, help="GPTQ group size")
    parser.add_argument("-W", "--weight_bits", type=int, default=4, help="GPTQ weight bits")
    args = parser.parse_args()

    M, K, block, gs, wb = args.M, args.K, args.block, args.group_size, args.weight_bits
    mode = "pcie" if args.hardware else "cmodel"

    assert wb == 4, "Current GPTQ test only covers 4-bit packed weights"
    assert M % block == 0, f"M={M} must be divisible by block={block}"
    assert K % block == 0, f"K={K} must be divisible by block={block}"
    assert block % gs == 0, f"block={block} must be divisible by group_size={gs}"

    tiles_m = M // block
    tiles_k = K // block
    total_tiles = tiles_m * tiles_k * 2 + tiles_m
    print(
        f"Testing M={M} K={K} block={block} gs={gs} wb={wb} "
        f"tiles={tiles_m}x{tiles_k} total~{total_tiles} mode={mode}"
    )

    try:
        kernel_func = llama_mlp_gptq(
            M=M,
            K=K,
            group_size=gs,
            weight_bits=wb,
            block_M=block,
            block_K=block,
            batch_size=1,
        )
        print("Compiling...")
        kernel = tilelang.compile(kernel_func, out_idx=[12], target="tpu", mode=mode)
    except Exception as e:
        print(f"Compilation failed: {e}")
        return

    gate_w, gate_zp, gate_scale = make_fake_gptq_weight(K, M, gs, seed=0)
    up_w, up_zp, up_scale = make_fake_gptq_weight(K, M, gs, seed=1)
    down_w, down_zp, down_scale = make_fake_gptq_weight(M, K, gs, seed=2)

    input_t = torch.randn(1, M, dtype=torch.float16) * 0.1
    output = torch.zeros(1, M, dtype=torch.float16)

    print("Running...")
    gs_tensor = torch.tensor([gs], dtype=torch.int32)
    wb_tensor = torch.tensor([wb], dtype=torch.int32)
    kernel(
        input_t,
        up_w, up_zp, up_scale,
        gate_w, gate_zp, gate_scale,
        down_w, down_zp, down_scale,
        gs_tensor, wb_tensor,
        output,
    )

    ref = reference_llama_mlp_gptq(
        input_t,
        up_w, up_zp, up_scale,
        gate_w, gate_zp, gate_scale,
        down_w, down_zp, down_scale,
        group_size=gs,
    )

    diff = (ref - output).float()
    valid = diff[~diff.isnan() & ~diff.isinf()]
    n_valid = valid.numel()
    total = diff.numel()

    print("\n=== Results ===")
    print(f"Valid elements: {n_valid}/{total}")
    if n_valid > 0:
        match = (valid.abs() < 1.0).sum().item()
        print(f"Match (atol=1.0): {match}/{n_valid} ({100 * match / n_valid:.1f}%)")
        print(f"Max diff: {valid.abs().max().item():.6f}")
        print(f"Avg diff: {valid.abs().mean().item():.6f}")

    print(f"TPU[0,:6]:  {output[0, :6].tolist()}")
    print(f"REF[0,:6]:  {ref[0, :6].tolist()}")

    if n_valid == total and torch.allclose(ref.float(), output.float(), atol=1.0, rtol=0.1):
        print("\n*** PASS ***")
    else:
        print("\n*** FAIL ***")


if __name__ == "__main__":
    main()
