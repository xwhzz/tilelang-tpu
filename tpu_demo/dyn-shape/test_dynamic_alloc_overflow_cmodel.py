"""CModel test for runtime Local Memory overflow guard."""

import ctypes
import os

import torch

import tilelang
import tilelang.language as T


M = T.symbolic("M")
N = 128


@T.prim_func
def overflow_alloc_kernel(
    X: T.Tensor((M, N), "float32"),
    Y: T.Tensor((M, N), "float32"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((M, N), "float32")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])


def capture_c_stdout(fn):
    libc = ctypes.CDLL(None)
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        ret = fn()
        libc.fflush(None)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(write_fd)
    output = os.read(read_fd, 1 << 20).decode("utf-8", errors="replace")
    os.close(read_fd)
    return ret, output


kernel = tilelang.compile(
    overflow_alloc_kernel,
    out_idx=[1],
    target="tpu",
    mode="cmodel",
)

M_val = 1024
x = torch.randn(M_val, N).float()
y = torch.zeros(M_val, N).float()

ret, output = capture_c_stdout(lambda: kernel(x, y))
print(output)

assert ret != 0, f"Expected Local Memory overflow to fail, got {ret}"
assert "Local Memory overflow" in output, f"Expected overflow diagnostic. Got:\n{output}"

print("\n" + "=" * 60)
print("DYNAMIC ALLOC OVERFLOW CMODEL TEST: PASSED")
print("=" * 60)
