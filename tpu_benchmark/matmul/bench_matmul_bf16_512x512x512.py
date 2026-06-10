"""
Benchmark: Matmul BF16 512x512x512 — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N, K = 512, 512, 512
BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
ATOL, RTOL = 1e-1, 1e-1


def tl_matmul_lowp(M, N, K, block_M, block_N, block_K):
    dtype = "bfloat16"
    accum_dtype = "bfloat16"

    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), "float32")
            C_shared_ori = T.alloc_shared((block_M, block_N), accum_dtype)

            T.fill(C_shared, T.float32(0))
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C_shared_ori)
            T.copy(C_shared_ori, C[by * block_M, bx * block_N])

    return main_kernel_inner


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(M, N, dtype=torch.bfloat16)
    kernel_func(a, b, c)
    correct = torch.allclose(c.float(), ref.float(), atol=ATOL, rtol=RTOL)
    max_diff = (c.float() - ref.float()).abs().max().item()
    avg_diff = (c.float() - ref.float()).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    return correct


def main():
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    ref = torch.matmul(a, b).bfloat16()

    print("=" * 60)
    print(f"Matmul BF16  M={M} N={N} K={K}  block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print("=" * 60)

    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_matmul_lowp(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel
        pl_path = os.path.join(BENCH_DIR, "pl", "matmul_bf16_512x512x512.pl")
        arg_specs = [((M, K), torch.bfloat16), ((K, N), torch.bfloat16), ((M, N), torch.bfloat16)]
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
