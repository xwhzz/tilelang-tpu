import tilelang
import tilelang.language as T
import torch

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


kernel = tilelang.compile(dynamic_copy_kernel, out_idx=[1], target="tpu")

# ── Test multiple dynamic shapes ──
for M_val in [64, 128, 256]:
    x = torch.randn(M_val, N_val, dtype=torch.float16)
    y = torch.zeros(M_val, N_val, dtype=torch.float16)
    y_ref = torch.zeros(M_val, N_val, dtype=torch.float16)
    y_ref[0, :] = x[0, :]

    kernel(x, y)

    diff = (y - y_ref).abs()
    ok = torch.allclose(y, y_ref, atol=1e-3)
    print(f"M={M_val:>4}: {'PASS' if ok else 'FAIL (max_diff=' + str(diff.max().item()) + ')'}")

print("done")
