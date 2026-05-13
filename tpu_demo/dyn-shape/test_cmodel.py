import tilelang
import tilelang.language as T
import torch

block_size = 128


def assert_close(name, actual, expected, tol=1e-2):
    max_diff = torch.max(torch.abs(actual.float() - expected.float())).item()
    assert max_diff < tol, f"{name} max diff {max_diff} exceeds tolerance {tol}"
    print(f"{name}: PASSED (max diff {max_diff:.6e})")


# @T.prim_func
# def static_copy(
#     X: T.Tensor((1, block_size), "float16"),
#     Y: T.Tensor((1, block_size), "float16"),
# ):
#     with T.Kernel(1, 1, is_cpu=True) as (bx, by):
#         local = T.alloc_shared((1, block_size), "float16")
#         T.ppl_copy(X[0, 0], local)
#         T.ppl_copy(local, Y[0, 0])


# kernel = tilelang.compile(static_copy, out_idx=-1, target="tpu", mode="cmodel")

# a = torch.randn(1, block_size).half()
# b = torch.zeros(1, block_size).half()
# kernel(a, b)
# assert_close("static copy", b, a)

M = T.symbolic("M")
N = 128


@T.prim_func
def dynamic_copy(
    X: T.Tensor((M, N), "float16"),
    Y: T.Tensor((M, N), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((M, N), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])


kernel2 = tilelang.compile(dynamic_copy, out_idx=-1, target="tpu", mode="cmodel")

# Test with M=3
a2 = torch.randn(3, N).half()
b2 = torch.zeros(3, N).half()
kernel2(a2, b2)
assert_close("dynamic copy M=3", b2, a2)

# Test with a different M value to verify dynamic shape works
a3 = torch.randn(7, N).half()
b3 = torch.zeros(7, N).half()
kernel2(a3, b3)
assert_close("dynamic copy M=7", b3, a3)

print("\n" + "=" * 60)
print("ALL CMODEL TESTS PASSED")
print("=" * 60)
