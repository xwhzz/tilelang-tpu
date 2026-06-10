"""
Benchmark: Elementwise Add FP32 512x512 — tilelang vs PPL
"""

import os, sys, torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_ROOT = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
import tilelang.language as T

M, N = 512, 512
BLOCK_M, BLOCK_N = 32, 32
ATOL, RTOL = 1e-5, 1e-5


def tl_add(M, N, block_M, block_N, dtype="float32"):
    @T.prim_func
    def main_kernel_inner(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            T.add(C_shared, A_shared, B_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main_kernel_inner


def run_and_check(name, kernel_func, a, b, ref):
    c = torch.zeros(M, N, dtype=torch.float32)
    kernel_func(a, b, c)
    correct = torch.allclose(c, ref, atol=ATOL, rtol=RTOL)
    max_diff = (c - ref).abs().max().item()
    avg_diff = (c - ref).abs().mean().item()
    print(f"  [{name}] correct={correct}  max_diff={max_diff:.6f}  avg_diff={avg_diff:.8f}")
    return correct


def main():
    a = torch.randn(M, N, dtype=torch.float32)
    b = torch.randn(M, N, dtype=torch.float32)
    ref = a + b

    print("=" * 60)
    print(f"Elementwise Add FP32  M={M} N={N}  block={BLOCK_M}x{BLOCK_N}")
    print("=" * 60)

    print("\n--- tilelang ---")
    tl_kernel = tilelang.compile(
        tl_add(M, N, BLOCK_M, BLOCK_N), out_idx=-1, target="tpu")
    run_and_check("tilelang", tl_kernel, a, b, ref)

    print("\n--- PPL ---")
    try:
        from ppl_utils import compile_ppl_kernel, generate_pl
        pl_path = generate_pl("add", "float32", {"M": M, "N": N})
        arg_specs = [((M, N), torch.float32)] * 3
        ppl_forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=[2])
        run_and_check("PPL", ppl_forward, a, b, ref)
    except Exception as e:
        print(f"  PPL failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
