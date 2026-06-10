"""
Benchmark: RMSNorm FP16 128x128 — tilelang (optimized) vs PPL

Optimization: eliminates A_pow2 buffer (square A_fp32 in-place, reload from GMEM).
This frees SRAM to double blk_m at N>=512, halving outer loop iterations.
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = 128, 128
BLK_M = 64
ATOL, RTOL = 5e-2, 5e-2


def tl_rmsnorm_lowp(M, N, blk_m):
    dtype = "float16"

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)
        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared     = T.alloc_shared((blk_m, N), dtype)
            A_shared_fp32= T.alloc_shared((blk_m, N), "float32")
            A_powsum     = T.alloc_shared((blk_m, 1), "float32")

            T.copy(A[bx * blk_m, 0], A_shared)
            T.copy(A_shared, A_shared_fp32)
            # Square in-place (fp32): A_fp32 = A^2 (destroys A_fp32)
            T.mul(A_shared_fp32, A_shared_fp32, A_shared_fp32)
            T.reduce_sum(A_shared_fp32, A_powsum, dim=1)
            T.mul_C(A_powsum, A_powsum, reciprocal_N)
            T.add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.rsqrt(A_powsum, A_powsum)
            # Reload cast to fp32, then scale, then cast back
            T.copy(A[bx * blk_m, 0], A_shared)
            T.copy(A_shared, A_shared_fp32)
            T.mul(A_shared_fp32, A_shared_fp32, A_powsum)
            T.copy(A_shared_fp32, A_shared)
            T.copy(A_shared, B[bx * blk_m, 0])

    return main_kernel_inner


def torch_ref(a, eps=1e-12):
    a_fp32 = a.float()
    ref_fp32 = a_fp32 / torch.sqrt(torch.mean(a_fp32 * a_fp32, dim=1, keepdim=True) + eps)
    return ref_fp32.to(a.dtype)


def run_and_check(name, kernel_func, a, ref):
    b = torch.zeros(M, N, dtype=torch.float16)
    kernel_func(a, b)
    correct = torch.allclose(b.float(), ref.float(), atol=ATOL, rtol=RTOL)
    max_diff = (b.float() - ref.float()).abs().max().item()
    avg_diff = (b.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    return correct


def main():
    a = torch.randn(M, N, dtype=torch.float16)
    ref = torch_ref(a)

    print("=" * 60)
    print(f"RMSNorm FP16 (opt)  M={M} N={N}  blk_m={BLK_M}")
    print("=" * 60)

    print("\n--- tilelang (optimized) ---")
    tl_kernel = tilelang.compile(
        tl_rmsnorm_lowp(M, N, BLK_M), out_idx=-1, target="tpu")
    run_and_check("tilelang-opt", tl_kernel, a, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "pl", "rmsnorm_fp16_128x128.pl")
        arg_specs = [((M, N), torch.float16), ((M, N), torch.float16)]
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[1])
        run_and_check("PPL", ppl_forward, a, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
