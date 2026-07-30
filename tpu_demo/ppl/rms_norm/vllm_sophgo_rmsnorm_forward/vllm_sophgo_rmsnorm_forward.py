# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os

import tilelang
import tilelang.language as T
import torch

def vllm_sophgo_rmsnorm_forward(
    M,
    N,
    blk_m=None,
    blk_n=None,
    dtype="float16",
):
    """RMSNorm with the torch.ops.my_ops.rmsnorm_forward argument order."""
    blk_m = blk_m or 1
    blk_n = blk_n or N

    @T.prim_func
    def main_kernel_inner(
        HiddenStates: T.Tensor((M, 1, 1, N), dtype),
        Weight: T.Tensor((1, 1, 1, N), dtype),
        Bias: T.Tensor((1, 1, 1, N), dtype),
        Output: T.Tensor((M, 1, 1, N), dtype),
        Axis: T.int32,
        Epsilon: T.float32,
    ):
        reciprocal_n = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            hidden_shared = T.alloc_shared(
                (blk_m, 1, 1, blk_n), dtype
            )
            hidden_fp32 = T.alloc_shared(
                (blk_m, 1, 1, blk_n), "float32"
            )
            hidden_squared = T.alloc_shared(
                (blk_m, 1, 1, blk_n), "float32"
            )
            square_sum = T.alloc_shared(
                (blk_m, 1, 1, 1), "float32"
            )
            tile_square_sum = T.alloc_shared(
                (blk_m, 1, 1, 1), "float32"
            )
            weight_shared = T.alloc_shared((1, 1, 1, blk_n), dtype)

            T.ppl_copy(
                HiddenStates[
                    bx * blk_m:(bx + 1) * blk_m,
                    0:1,
                    0:1,
                    0:blk_n,
                ],
                hidden_shared,
            )
            T.ppl_copy(hidden_shared, hidden_fp32)
            T.ppl_mul(hidden_squared, hidden_fp32, hidden_fp32)
            T.ppl_reduce_sum(hidden_squared, square_sum, dim=1)

            for k in range(1, N // blk_n):
                T.ppl_copy(
                    HiddenStates[
                        bx * blk_m:(bx + 1) * blk_m,
                        0:1,
                        0:1,
                        k * blk_n:(k + 1) * blk_n,
                    ],
                    hidden_shared,
                )
                T.ppl_copy(hidden_shared, hidden_fp32)
                T.ppl_mul(hidden_squared, hidden_fp32, hidden_fp32)
                T.ppl_reduce_sum(hidden_squared, tile_square_sum, dim=1)
                T.ppl_add(square_sum, square_sum, tile_square_sum)

            T.ppl_mul_C(square_sum, square_sum, reciprocal_n)
            T.ppl_add_C(square_sum, square_sum, Epsilon)
            T.ppl_rsqrt(square_sum, square_sum)

            for k in range(N // blk_n):
                T.ppl_copy(
                    HiddenStates[
                        bx * blk_m:(bx + 1) * blk_m,
                        0:1,
                        0:1,
                        k * blk_n:(k + 1) * blk_n,
                    ],
                    hidden_shared,
                )
                T.ppl_copy(hidden_shared, hidden_fp32)
                T.ppl_mul(hidden_fp32, hidden_fp32, square_sum)
                T.ppl_copy(hidden_fp32, hidden_shared)
                T.ppl_copy(
                    Weight[
                        0:1,
                        0:1,
                        0:1,
                        k * blk_n:(k + 1) * blk_n,
                    ],
                    weight_shared,
                )
                T.ppl_mul(hidden_shared, hidden_shared, weight_shared)
                T.ppl_copy(
                    hidden_shared,
                    Output[
                        bx * blk_m:(bx + 1) * blk_m,
                        0:1,
                        0:1,
                        k * blk_n:(k + 1) * blk_n,
                    ],
                )

    return main_kernel_inner


def build_kernel(m, n, dtype="float16", mode="pcie"):
    return tilelang.compile(
        vllm_sophgo_rmsnorm_forward(m, n, dtype=dtype),
        out_idx=3,
        target="tpu",
        mode=mode,
    )


if __name__ == "__main__":
    m = int(os.environ.get("RMS_M", "128"))
    n = int(os.environ.get("RMS_N", "128"))
    eps = float(os.environ.get("RMS_EPS", "1e-6"))
    mode = os.environ.get("RMS_MODE", "pcie")
    device = os.environ.get("RMS_DEVICE", "cpu" if mode == "cmodel" else "tpu:0")

    if device.startswith("tpu"):
        import torch_tpu  # noqa: F401

    kernel = build_kernel(m, n, mode=mode)
    hidden_states = torch.randn(m, n, dtype=torch.float16, device=device)
    weight = torch.randn(n, dtype=torch.float16, device=device)
    output = torch.empty_like(hidden_states)

    kernel(
        hidden_states,
        weight,
        None,
        output,
        hidden_states.dim() - 1,
        eps,
    )

    ref = hidden_states.float()
    ref = ref * torch.rsqrt(torch.mean(ref * ref, dim=-1, keepdim=True) + eps)
    ref = (ref * weight.float()).to(hidden_states.dtype)

    max_diff = torch.max(torch.abs(ref.float() - output.float()))
    print(f"max diff: {max_diff}")
    print(f"check close: {torch.allclose(output.float(), ref.float(), atol=5e-2, rtol=5e-2)}")
