# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import time

import torch

import tilelang
import tilelang.language as T


def choose_blk_m(m, max_blk_m=128):
    blk_m = min(m, max_blk_m)
    while m % blk_m != 0:
        blk_m -= 1
    return blk_m

def vllm_sophgo_rmsnorm_forward(M, N, blk_m=32, eps=1e-6, dtype="float16"):
    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((1, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)

        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            X_shared = T.alloc_shared((blk_m, N), dtype)
            X_fp32 = T.alloc_shared((blk_m, N), "float32")
            X_pow2 = T.alloc_shared((blk_m, N), "float32")
            X_powsum = T.alloc_shared((blk_m, 1), "float32")
            Weight_shared = T.alloc_shared((1, N), dtype)

            T.ppl_copy(X[bx * blk_m:(bx + 1) * blk_m, :], X_shared)
            T.ppl_copy(X_shared, X_fp32)
            T.ppl_mul(X_pow2, X_fp32, X_fp32)
            T.ppl_reduce_sum(X_pow2, X_powsum, dim=1)
            T.ppl_mul_C(X_powsum, X_powsum, reciprocal_N)
            T.ppl_add_C(X_powsum, X_powsum, T.float32(eps))
            T.ppl_rsqrt(X_powsum, X_powsum)
            T.ppl_mul(X_fp32, X_fp32, X_powsum)
            T.ppl_copy(X_fp32, X_shared)
            T.ppl_copy(Weight[0:1, :], Weight_shared)
            T.ppl_mul(X_shared, X_shared, Weight_shared)
            T.ppl_copy(X_shared, Output[bx * blk_m:(bx + 1) * blk_m, :])

    return main_kernel_inner


def copy_forward(M, N, blk_m=32, dtype="float16"):
    @T.prim_func
    def main_kernel_inner(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((1, N), dtype),
        Output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            X_shared = T.alloc_shared((blk_m, N), dtype)

            T.ppl_copy(X[bx * blk_m:(bx + 1) * blk_m, :], X_shared)
            T.ppl_copy(X_shared, Output[bx * blk_m:(bx + 1) * blk_m, :])

    return main_kernel_inner


def build_kernel(m, n, eps=1e-6, dtype="float16", kernel_mode="rmsnorm"):
    blk_m = choose_blk_m(m)
    if kernel_mode == "copy":
        program = copy_forward(m, n, blk_m, dtype)
    else:
        program = vllm_sophgo_rmsnorm_forward(m, n, blk_m, eps, dtype)

    return tilelang.compile(
        program,
        out_idx=2,
        target="tpu",
        mode="cmodel",
    )


def run_verify():
    m = 128
    n = 128
    eps = 1e-6
    dtype = "float16"
    input_mode = os.environ.get("RMS_INPUT_MODE", "random")
    kernel_mode = os.environ.get("RMS_KERNEL_MODE", "rmsnorm")
    output_init = os.environ.get("RMS_OUTPUT_INIT", "zero")

    kernel = build_kernel(m, n, eps=eps, dtype=dtype, kernel_mode=kernel_mode)

    if input_mode == "ones":
        x = torch.ones(m, n).half()
        weight = torch.ones(1, n).half()
    else:
        torch.manual_seed(0)
        x = torch.randn(m, n).half()
        weight = torch.randn(1, n).half()
    if output_init == "sentinel":
        output = torch.full((m, n), 7.0).half()
    else:
        output = torch.zeros(m, n).half()

    ret = kernel(x, weight, output)
    print(f"[rmsnorm] kernel ret: {ret}")

    if kernel_mode == "copy":
        ref = x
    else:
        ref = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
        ref = (ref * weight.float()).to(x.dtype)

    diff = ref.float() - output.float()
    max_diff = torch.max(torch.abs(diff))
    avg_diff = torch.mean(torch.abs(diff))
    allclose = torch.allclose(output.float(), ref.float(), atol=5e-2, rtol=5e-2)
    output_nonzero = torch.count_nonzero(output).item()
    ref_nonzero = torch.count_nonzero(ref).item()

    print(f"[rmsnorm] input mode: {input_mode}")
    print(f"[rmsnorm] kernel mode: {kernel_mode}")
    print(f"[rmsnorm] output init: {output_init}")
    print(f"[rmsnorm] input first row first 8: {x[0, :8].float().tolist()}")
    print("[rmsnorm] output:")
    print(output)
    print("[rmsnorm] ref:")
    print(ref)
    print(f"[rmsnorm] output first row first 8: {output[0, :8].float().tolist()}")
    print(f"[rmsnorm] ref first row first 8: {ref[0, :8].float().tolist()}")
    print(f"[rmsnorm] output nonzero: {output_nonzero} / {output.numel()}")
    print(f"[rmsnorm] ref nonzero: {ref_nonzero} / {ref.numel()}")
    print(f"[rmsnorm] max diff: {max_diff}")
    print(f"[rmsnorm] avg diff: {avg_diff}")
    print(f"[rmsnorm] check close: {allclose}")

    return kernel, x, weight, output


def run_benchmark(kernel, x, weight, output, warmup=10, iters=50):
    for _ in range(warmup):
        kernel(x, weight, output)

    start = time.perf_counter()
    for _ in range(iters):
        kernel(x, weight, output)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / iters
    print(f"[rmsnorm] warmup: {warmup}, iters: {iters}, avg latency: {avg_ms:.3f} ms")


def main():
    kernel, x, weight, output = run_verify()
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    iters = int(os.environ.get("BENCH_ITERS", "50"))
    if iters > 0:
        run_benchmark(kernel, x, weight, output, warmup=warmup, iters=iters)


if __name__ == "__main__":
    main()
