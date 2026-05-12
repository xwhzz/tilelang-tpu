"""Lowering test for symbolic Local Memory allocation.

This verifies that AddressAssign accepts an alloc_shared buffer whose shape
contains a TIR symbolic variable, and that CodeGenTileLangPPL can emit the
local tensor metadata without forcing the symbolic dimension to IntImm.
"""

import tilelang
import tilelang.language as T


N = T.symbolic("N")


@T.prim_func
def dynamic_alloc_kernel(
    X: T.Tensor((1, N), "float16"),
    Y: T.Tensor((1, N), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, N), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])


mod = tilelang.lower(dynamic_alloc_kernel)
src = mod.kernel_source

print("=" * 60)
print("DYNAMIC LOCAL ALLOC C SOURCE")
print("=" * 60)
print(src)

assert "int N" in src, "Expected symbolic dim N in the kernel signature"
assert "N" in src, "Expected symbolic dim N in generated code"
assert "__ppl_tensor_info local" in src, "Expected local tensor metadata"
assert ".addr = " in src, "Expected local tensor address field"
assert ".size = " in src, "Expected local tensor size field"
assert "tpu_aligned_stride" in src, "Expected local stride setup"
assert "tpu_gdma_cpy_S2L" in src, "Expected lowered copy into symbolic local buffer"
assert "tpu_gdma_cpy_L2S" in src, "Expected lowered copy out of symbolic local buffer"

print("\nDYNAMIC LOCAL ALLOC TEST: PASSED")
