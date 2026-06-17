import tilelang
import tilelang.language as T

# Static test: simple copy kernel with all-static shapes
@T.prim_func
def main_kernel_inner(
    X: T.Tensor((64, 128), "float16"),
    Y: T.Tensor((64, 128), "float16"),
):
    with T.Kernel(1, 1, is_cpu=True) as (bx, by):
        local = T.alloc_shared((1, 128), "float16")
        T.ppl_copy(X[0, 0], local)
        T.ppl_copy(local, Y[0, 0])

mod = tilelang.lower(main_kernel_inner)
src = mod.kernel_source

# Check function signature has NO extra int dim params for static kernel
sig_line = [l for l in src.split('\n') if 'main_kernel_inner(' in l][0]
# The signature should be: void main_kernel_inner(global_addr_t v1, global_addr_t v2)
# For static kernels, no "int" should appear between the parens
params_part = sig_line.split("(")[1].split(")")[0]
print(f"Function signature: void main_kernel_inner({params_part})")
print(f"Has extra int dim params: {'int' in params_part}")
assert "int" not in params_part, f"Static kernel should NOT have extra int dim params! Got: {params_part}"
print("=== PASS: Static kernel unchanged ===")
