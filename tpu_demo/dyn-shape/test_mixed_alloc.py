"""Lowering test for mixed static and symbolic Local Memory allocations."""

import re

import tilelang
import tilelang.language as T


M = T.symbolic("M")
N = T.symbolic("N")


@T.prim_func
def mixed_alloc_kernel(
    X_dyn: T.Tensor((M, N), "float16"),
    Y_dyn: T.Tensor((M, N), "float16"),
    X_static: T.Tensor((1, 128), "float16"),
    Y_static: T.Tensor((1, 128), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        static_local = T.alloc_shared((1, 128), "float16")
        dynamic_local = T.alloc_shared((M, N), "float16")

        T.ppl_copy(X_static[0, 0], static_local)
        T.ppl_copy(static_local, Y_static[0, 0])
        T.ppl_copy(X_dyn[0, 0], dynamic_local)
        T.ppl_copy(dynamic_local, Y_dyn[0, 0])


def tensor_info(src: str, name: str) -> str:
    match = re.search(rf"__ppl_tensor_info\s+{name}\s*=\s*\{{[^;]*\}};", src)
    assert match is not None, f"Expected tensor info for {name}. Got:\n{src}"
    return match.group(0)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


mod = tilelang.lower(mixed_alloc_kernel)
src = mod.kernel_source

print("=" * 60)
print("MIXED STATIC/DYNAMIC LOCAL ALLOC C SOURCE")
print("=" * 60)
print(src)

static_info = tensor_info(src, "static_local")
dynamic_info = tensor_info(src, "dynamic_local")
static_compact = compact(static_info)
dynamic_compact = compact(dynamic_info)

assert "int M" in src, "Expected symbolic dim M in the kernel signature/API"
assert "int N" in src, "Expected symbolic dim N in the kernel signature/API"
assert ".shape={1,1,1,128}" in static_compact, \
    f"Expected static local shape to stay static. Got:\n{static_info}"
assert ".shape={1,M,1,N}" in dynamic_compact, \
    f"Expected dynamic local shape to keep M/N. Got:\n{dynamic_info}"
assert re.search(r"\.addr\s*=\s*\d+", static_info), \
    f"Expected static local addr to be a constant. Got:\n{static_info}"
assert re.search(r"\.addr\s*=\s*[1-9]\d*", dynamic_info), \
    f"Expected dynamic local addr to be placed after static LM. Got:\n{dynamic_info}"
assert re.search(r"\.size\s*=\s*\d+", static_info), \
    f"Expected static local size metadata. Got:\n{static_info}"
assert "M" in dynamic_info and "N" in dynamic_info and ".size" in dynamic_info, \
    f"Expected dynamic local size metadata to reference M/N. Got:\n{dynamic_info}"
assert "tpu_gdma_cpy_S2L" in src, "Expected lowered copies into local buffers"
assert "tpu_gdma_cpy_L2S" in src, "Expected lowered copies out of local buffers"

print("\nMIXED STATIC/DYNAMIC LOCAL ALLOC TEST: PASSED")
