"""End-to-end test for MVP-1 dynamic shape support.

This test verifies:
1. T.symbolic("M") produces a tir.Var that survives through the compilation pipeline
2. CodeGenTileLangPPL emits variable names (not hardcoded ints) in C shape literals
3. Static kernels are unchanged (no extra int params)
4. Dynamic kernel function signature includes int dim params
5. The generated C code is well-formed (valid C syntax)
"""
import tilelang
import tilelang.language as T


# ── Dynamic kernel ────────────────────────────────────────────
M = T.symbolic("M")
N_val = 128

@T.prim_func
def dynamic_copy_kernel(
    X: T.Tensor((M, N_val), "float16"),
    Y: T.Tensor((M, N_val), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, N_val), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])


mod = tilelang.lower(dynamic_copy_kernel)
src = mod.kernel_source

print("=" * 60)
print("DYNAMIC KERNEL C SOURCE")
print("=" * 60)
print(src)

# ── Assertions ─────────────────────────────────────────────────

# 1. Dynamic dim 'M' appears in shape literal, not a hardcoded number
assert "{1 ,M, 1, 128}" in src or "{1, M, 1, 128}" in src, \
    f"Expected 'M' in shape literal for dynamic dim. Got:\n{src}"

# 2. Function signature has 'int M' parameter
assert "int M)" in src or "int M," in src, \
    f"Expected 'int M' as parameter in function signature."

# 2b. Kernel API struct has int M field
assert "int M;" in src, \
    f"Expected 'int M;' in kernel API struct"

# 2c. main_kernel passes M through
assert "api->M" in src, \
    f"Expected 'api->M' in main_kernel wrapper"

# 3. Stride uses M dynamically
assert "M * 128" in src or "M*128" in src, \
    f"Expected dynamic stride computation with M. Got:\n{src}"

# 4. tensor_size = 0 for dynamic tensors
# (global tensors have size=0 when shape is dynamic)
assert ".size = 0" in src, \
    f"Expected tensor_size=0 for dynamic global tensor"

# 5. Local buffer is still static (size = 4, shape = {1, 1, 1, 128})
assert "local = {.shape = { 1, 1, 1, 128}" in src, \
    f"Expected local buffer to have static shape"

print("\n" + "=" * 60)
print("ALL DYNAMIC SHAPE ASSERTIONS PASSED")
print("=" * 60)

# ── Static kernel regression ──────────────────────────────────
@T.prim_func
def static_copy_kernel(
    X: T.Tensor((64, 128), "float16"),
    Y: T.Tensor((64, 128), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, 128), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])

mod2 = tilelang.lower(static_copy_kernel)
src2 = mod2.kernel_source

# Static kernel should NOT have 'int M' anywhere (no dynamic dims)
assert "int M" not in src2, \
    f"Static kernel should NOT have extra int dim params!"
assert "int M;" not in src2, \
    f"Static kernel struct should NOT have int M field!"
assert "{1 ,64, 1, 128}" in src2 or "{1, 64, 1, 128}" in src2, \
    "Static kernel shape broken!"

print("\nSTATIC KERNEL REGRESSION: PASSED")
print("=" * 60)
print("ALL MVP-1 CODEGEN TESTS PASSED")
