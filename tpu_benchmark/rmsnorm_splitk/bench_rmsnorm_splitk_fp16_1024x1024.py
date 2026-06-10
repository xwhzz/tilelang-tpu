"""
Benchmark: RMSNorm-SplitK FP16 1024x1024 — tilelang (splitk) vs PPL
SplitK: two-pass K-tiled kernel — blk_k=32 regardless of N, no SRAM overflow.
Cast to fp32 for accumulation in both passes.
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = 1024, 1024
BLK_M, BLK_K = 32, 32
ATOL, RTOL = 5e-2, 5e-2


def tl_rmsnorm_splitk_lowp(M, N, blk_m, blk_k):
    dtype = "float16"

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        reciprocal_N = T.float32(1.0 / N)
        with T.Kernel(T.ceildiv(M, blk_m), is_cpu=True) as (bx,):
            A_shared    = T.alloc_shared((blk_m, blk_k), dtype)
            A_fp32      = T.alloc_shared((blk_m, blk_k), "float32")
            A_pow2      = T.alloc_shared((blk_m, blk_k), "float32")
            A_powsum    = T.alloc_shared((blk_m, 1), "float32")
            A_temp      = T.alloc_shared((blk_m, 1), "float32")

            T.fill(A_powsum, T.float32(0.0))
            num_k_step = T.ceildiv(N, blk_k)

            # Pass 1: accumulate sum of squares
            for k in T.Pipelined(num_k_step, num_stages=0):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                T.copy(A_shared, A_fp32)
                T.mul(A_pow2, A_fp32, A_fp32)
                T.reduce_sum(A_pow2, A_temp, dim=1)
                T.add(A_powsum, A_powsum, A_temp)

            T.mul_C(A_powsum, A_powsum, reciprocal_N)
            T.add_C(A_powsum, A_powsum, T.float32(1e-12))
            T.rsqrt(A_powsum, A_powsum)

            # Pass 2: normalize and write back
            for k in T.Pipelined(num_k_step, num_stages=0):
                block_k_idx = num_k_step - 1 - k
                T.copy(A[bx * blk_m, block_k_idx * blk_k], A_shared)
                T.copy(A_shared, A_fp32)
                T.mul(A_fp32, A_fp32, A_powsum)
                T.copy(A_fp32, A_shared)
                T.copy(A_shared, B[bx * blk_m, block_k_idx * blk_k])

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
    print(f"RMSNorm-SplitK FP16  M={M} N={N}  blk_m={BLK_M} blk_k={BLK_K}")
    print("=" * 60)

    print("\n--- tilelang splitk ---")
    tl_kernel = tilelang.compile(
        tl_rmsnorm_splitk_lowp(M, N, BLK_M, BLK_K), out_idx=-1, target="tpu")
    run_and_check("tilelang-splitk", tl_kernel, a, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "..", "rmsnorm", "pl", "rmsnorm_fp16_1024x1024.pl")
        arg_specs = [((M, N), torch.float16), ((M, N), torch.float16)]
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[1])
        run_and_check("PPL", ppl_forward, a, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
