"""CModel test for mixed static and dynamic Local Memory allocations."""

import torch

import tilelang
import tilelang.language as T


M = T.symbolic("M")
N = 128


def assert_close(name, actual, expected, tol=1e-5):
    diff = torch.abs(actual.float() - expected.float())
    max_diff = torch.max(diff).item()
    avg_diff = torch.mean(diff).item()
    print(f"{name}: max diff {max_diff:.6e}, avg diff {avg_diff:.6e}")
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"


@T.prim_func
def mixed_alloc_kernel(
    X_dyn: T.Tensor((M, N), "float32"),
    Y_dyn: T.Tensor((M, N), "float32"),
    X_static: T.Tensor((1, N), "float32"),
    Y_static: T.Tensor((1, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        static_local = T.alloc_shared((1, N), "float32")
        dynamic_local = T.alloc_shared((M, N), "float32")

        T.ppl_copy(X_static[0, 0], static_local)
        T.ppl_copy(static_local, Y_static[0, 0])
        T.ppl_copy(X_dyn[0, 0], dynamic_local)
        T.ppl_copy(dynamic_local, Y_dyn[0, 0])


kernel = tilelang.compile(
    mixed_alloc_kernel,
    out_idx=[1, 3],
    target="tpu",
    mode="cmodel",
)

torch.manual_seed(0)

for M_val in [3, 7]:
    x_dyn = torch.randn(M_val, N).float()
    y_dyn = torch.zeros(M_val, N).float()
    x_static = torch.randn(1, N).float()
    y_static = torch.zeros(1, N).float()

    kernel(x_dyn, y_dyn, x_static, y_static)

    assert_close(f"dynamic local copy M={M_val}", y_dyn, x_dyn)
    assert_close(f"static local copy M={M_val}", y_static, x_static)

print("\n" + "=" * 60)
print("MIXED STATIC/DYNAMIC ALLOC CMODEL TEST: PASSED")
print("=" * 60)
