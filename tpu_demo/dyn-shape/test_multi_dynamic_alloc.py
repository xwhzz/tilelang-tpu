"""Lowering test for stacked symbolic Local Memory allocations."""

import re

import tilelang
import tilelang.language as T


M = T.symbolic("M")
K = T.symbolic("K")
N = 128


@T.prim_func
def multi_dynamic_alloc_kernel(
    X0: T.Tensor((M, N), "float16"),
    X1: T.Tensor((K, N), "float16"),
    X2: T.Tensor((M, 64), "float16"),
    Y0: T.Tensor((M, N), "float16"),
    Y1: T.Tensor((K, N), "float16"),
    Y2: T.Tensor((M, 64), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        dyn0 = T.alloc_shared((M, N), "float16")
        dyn1 = T.alloc_shared((K, N), "float16")
        dyn2 = T.alloc_shared((M, 64), "float16")

        T.ppl_copy(X0[0, 0], dyn0)
        T.ppl_copy(X1[0, 0], dyn1)
        T.ppl_copy(X2[0, 0], dyn2)
        T.ppl_copy(dyn0, Y0[0, 0])
        T.ppl_copy(dyn1, Y1[0, 0])
        T.ppl_copy(dyn2, Y2[0, 0])


def tensor_info(src: str, name: str) -> str:
    match = re.search(rf"__ppl_tensor_info\s+{name}\s*=\s*\{{[^;]*\}};", src)
    assert match is not None, f"Expected tensor info for {name}. Got:\n{src}"
    return match.group(0)


def field(info: str, name: str, next_name: str) -> str:
    match = re.search(rf"\.{name}\s*=\s*(.*?),\s*\.{next_name}", info)
    assert match is not None, f"Expected field {name}. Got:\n{info}"
    return match.group(1)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


mod = tilelang.lower(multi_dynamic_alloc_kernel)
src = mod.kernel_source

print("=" * 60)
print("MULTI DYNAMIC LOCAL ALLOC C SOURCE")
print("=" * 60)
print(src)

dyn0_info = tensor_info(src, "dyn0")
dyn1_info = tensor_info(src, "dyn1")
dyn2_info = tensor_info(src, "dyn2")

dyn0_compact = compact(dyn0_info)
dyn1_compact = compact(dyn1_info)
dyn2_compact = compact(dyn2_info)

addr0 = compact(field(dyn0_info, "addr", "dtype"))
addr1 = compact(field(dyn1_info, "addr", "dtype"))
addr2 = compact(field(dyn2_info, "addr", "dtype"))

assert "int M" in src, "Expected symbolic dim M in the kernel signature/API"
assert "int K" in src, "Expected symbolic dim K in the kernel signature/API"
assert ".shape={1,M,1,128}" in dyn0_compact, \
    f"Expected dyn0 shape to keep M. Got:\n{dyn0_info}"
assert ".shape={1,K,1,128}" in dyn1_compact, \
    f"Expected dyn1 shape to keep K. Got:\n{dyn1_info}"
assert ".shape={1,M,1,64}" in dyn2_compact, \
    f"Expected dyn2 shape to keep M. Got:\n{dyn2_info}"

assert addr0 == "0", f"Expected first dynamic buffer at LM base 0. Got: {addr0}"
assert addr1 != "0" and "M" in addr1, \
    f"Expected second dynamic buffer addr to depend on dyn0 size. Got: {addr1}"
assert addr2 != "0" and "M" in addr2 and "K" in addr2, \
    f"Expected third dynamic buffer addr to depend on dyn0+dyn1 sizes. Got: {addr2}"
assert addr1 != addr2, \
    f"Expected each dynamic buffer to receive a distinct stacked addr. Got: {addr1}"
assert src.count("tpu_gdma_cpy_S2L") >= 3, "Expected copies into all dynamic buffers"
assert src.count("tpu_gdma_cpy_L2S") >= 3, "Expected copies out of all dynamic buffers"

print("\nMULTI DYNAMIC LOCAL ALLOC TEST: PASSED")
