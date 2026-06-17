"""CModel test for multiple stacked dynamic Local Memory allocations."""

import torch

import tilelang
import tilelang.language as T


M = T.symbolic("M")
K = T.symbolic("K")
N = 128


def make_input(rows, cols, base):
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) + base


def assert_close(name, actual, expected, tol=1e-5):
    diff = torch.abs(actual.float() - expected.float())
    max_diff = torch.max(diff).item()
    avg_diff = torch.mean(diff).item()
    print(f"{name}: max diff {max_diff:.6e}, avg diff {avg_diff:.6e}")
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"


@T.prim_func
def multi_dynamic_alloc_kernel(
    X0: T.Tensor((M, N), "float32"),
    X1: T.Tensor((K, N), "float32"),
    X2: T.Tensor((M, 64), "float32"),
    Y0: T.Tensor((M, N), "float32"),
    Y1: T.Tensor((K, N), "float32"),
    Y2: T.Tensor((M, 64), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        dyn0 = T.alloc_shared((M, N), "float32")
        dyn1 = T.alloc_shared((K, N), "float32")
        dyn2 = T.alloc_shared((M, 64), "float32")

        T.ppl_copy(X0[0, 0], dyn0)
        T.ppl_copy(X1[0, 0], dyn1)
        T.ppl_copy(X2[0, 0], dyn2)
        T.ppl_copy(dyn0, Y0[0, 0])
        T.ppl_copy(dyn1, Y1[0, 0])
        T.ppl_copy(dyn2, Y2[0, 0])


kernel = tilelang.compile(
    multi_dynamic_alloc_kernel,
    out_idx=[3, 4, 5],
    target="tpu",
    mode="cmodel",
)

for M_val, K_val in [(3, 5), (7, 2)]:
    x0 = make_input(M_val, N, 1000.0)
    x1 = make_input(K_val, N, 2000.0)
    x2 = make_input(M_val, 64, 3000.0)
    y0 = torch.zeros(M_val, N).float()
    y1 = torch.zeros(K_val, N).float()
    y2 = torch.zeros(M_val, 64).float()

    kernel(x0, x1, x2, y0, y1, y2)

    assert_close(f"dyn0 copy M={M_val}, K={K_val}", y0, x0)
    assert_close(f"dyn1 copy M={M_val}, K={K_val}", y1, x1)
    assert_close(f"dyn2 copy M={M_val}, K={K_val}", y2, x2)

print("\n" + "=" * 60)
print("MULTI DYNAMIC ALLOC CMODEL TEST: PASSED")
print("=" * 60)
