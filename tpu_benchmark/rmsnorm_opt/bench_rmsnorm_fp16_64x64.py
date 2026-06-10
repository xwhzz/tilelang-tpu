"""
Benchmark: optimized RMSNorm FP16 64x64 vs PPL.

Fixed TL variant: `reload_b64`.
Timing uses the TPU template default: 5 warmup runs, then 10 measured runs.
"""

import json
import os
import sys

import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
RMSNORM_DIR = os.path.join(BENCHMARK_ROOT, "rmsnorm")
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = 64, 64
BLK_M = 64
VARIANT = "reload_b64"
STRATEGY = "reload"
EPS = 1e-12
TORCH_DTYPE = torch.float16
TL_DTYPE = "float16"
DTYPE_NAME = "fp16"
ATOL, RTOL = 5e-2, 5e-2


def tl_rmsnorm_opt(M, N, blk_m):
    dtype = TL_DTYPE

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)
        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_shared_fp32 = T.alloc_shared((blk_m, N), "float32")
            A_powsum = T.alloc_shared((blk_m, 1), "float32")

            T.copy(A[bx * blk_m, 0], A_shared)
            T.copy(A_shared, A_shared_fp32)
            T.mul(A_shared_fp32, A_shared_fp32, A_shared_fp32)
            T.reduce_sum(A_shared_fp32, A_powsum, dim=1)
            T.mul_C(A_powsum, A_powsum, reciprocal_N)
            T.add_C(A_powsum, A_powsum, T.float32(EPS))
            T.rsqrt(A_powsum, A_powsum)
            T.copy(A[bx * blk_m, 0], A_shared)
            T.copy(A_shared, A_shared_fp32)
            T.mul(A_shared_fp32, A_shared_fp32, A_powsum)
            T.copy(A_shared_fp32, A_shared)
            T.copy(A_shared, B[bx * blk_m, 0])

    return main_kernel_inner


def torch_ref(a):
    a_fp32 = a.float()
    ref = a_fp32 / torch.sqrt(torch.mean(a_fp32 * a_fp32, dim=1, keepdim=True) + EPS)
    return ref.to(a.dtype)


def run_and_check(name, kernel_func, a, ref):
    out = torch.zeros(M, N, dtype=TORCH_DTYPE)
    kernel_func(a, out)
    if DTYPE_NAME == "fp32":
        correct = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
    else:
        correct = torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
    max_diff = (out.float() - ref.float()).abs().max().item()
    avg_diff = (out.float() - ref.float()).abs().mean().item()
    result = {
        "name": name,
        "correct": bool(correct),
        "max_diff": max_diff,
        "avg_diff": avg_diff,
    }
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    print("RESULT_JSON:" + json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
    return correct


def main():
    torch.manual_seed(0)
    a = torch.randn(M, N, dtype=TORCH_DTYPE)
    ref = torch_ref(a)

    print("=" * 72)
    print(f"RMSNorm FP16 optimized  M={M} N={N}  variant={VARIANT}  BLK_M={BLK_M}")
    print("=" * 72, flush=True)

    print("\n--- tilelang optimized ---")
    tl_kernel = tilelang.compile(tl_rmsnorm_opt(M, N, BLK_M), out_idx=-1, target="tpu")
    run_and_check("tilelang-opt", tl_kernel, a, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel

        pl_path = os.path.join(RMSNORM_DIR, "pl", f"rmsnorm_{DTYPE_NAME}_{M}x{N}.pl")
        arg_specs = [((M, N), TORCH_DTYPE), ((M, N), TORCH_DTYPE)]
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[1])
        run_and_check("PPL", ppl_forward, a, ref)
    except Exception as exc:
        print(f"  PPL failed: {exc}")
        raise

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
