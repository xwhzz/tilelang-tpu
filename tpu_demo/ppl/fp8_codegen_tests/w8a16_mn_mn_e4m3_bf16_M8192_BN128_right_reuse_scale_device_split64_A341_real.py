import tilelang
import tilelang.language as T
import torch


def ceildiv(a, b):
    return (a + b - 1) // b


def expand_scale_group_to_full(
    scale_group: torch.Tensor,
    N: int,
    K: int,
    group_N: int,
    group_K: int,
):
    """Reference-only full scale expansion."""
    return (
        scale_group
        .repeat_interleave(group_N, dim=0)
        .repeat_interleave(group_K, dim=1)
    )[:N, :K].contiguous()


def w8a16_mn_mn_right_reuse_bn128_scale_device_split64_A341_kernel(
    M=1024,
    K=128,
    N=128,
    block_M=128,
    block_N=128,
    block_K=128,
    split_N=64,
    group_N=128,
    group_K=128,
    dtype_left="bfloat16",
    dtype_right="e4m3_float8",
    dtype_scale="bfloat16",
    dtype_out="bfloat16",
    dtype_compute="bfloat16",
    accum_dtype="float32",
):
    if K != block_K:
        raise ValueError(f"A3.4.1 currently supports only K == block_K, got K={K}, block_K={block_K}")
    if block_N != 2 * split_N:
        raise ValueError(f"A3.4.1 expects block_N == 2 * split_N, got block_N={block_N}, split_N={split_N}")
    if block_N != group_N:
        raise ValueError(f"A3.4.1 prototype expects block_N == group_N, got block_N={block_N}, group_N={group_N}")

    K_GROUP = ceildiv(K, group_K)
    N_GROUP = ceildiv(N, group_N)
    BLOCK_K_GROUP = ceildiv(block_K, group_K)

    @T.prim_func
    def main_kernel_inner(
        G_left: T.Tensor((M, K), dtype_left),
        G_right_fp8: T.Tensor((N, K), dtype_right),
        G_scale_group: T.Tensor((N_GROUP, K_GROUP), dtype_scale),
        G_out: T.Tensor((M, N), dtype_out),
    ):
        with T.Kernel(T.ceildiv(N, block_N), 1, is_cpu=True) as (bx, _):
            right_fp8_0 = T.alloc_shared((split_N, block_K), dtype_right)
            right_fp8_1 = T.alloc_shared((split_N, block_K), dtype_right)

            right_bf16_0 = T.alloc_shared((split_N, block_K), dtype_compute)
            right_bf16_1 = T.alloc_shared((split_N, block_K), dtype_compute)

            scale_group_shared = T.alloc_shared((1, BLOCK_K_GROUP), dtype_scale)
            scale_kgroup_0 = T.alloc_shared((split_N, BLOCK_K_GROUP), dtype_scale)
            scale_kgroup_1 = T.alloc_shared((split_N, BLOCK_K_GROUP), dtype_scale)

            right_scaled_0 = T.alloc_shared((split_N, block_K), dtype_compute)
            right_scaled_1 = T.alloc_shared((split_N, block_K), dtype_compute)

            left_shared = T.alloc_shared((block_M, block_K), dtype_left)

            out_shared_0 = T.alloc_shared((block_M, split_N), accum_dtype)
            out_shared_1 = T.alloc_shared((block_M, split_N), accum_dtype)

            out_cast_0 = T.alloc_shared((block_M, split_N), dtype_out)
            out_cast_1 = T.alloc_shared((block_M, split_N), dtype_out)

            T.ppl_copy(G_right_fp8[bx * block_N, 0], right_fp8_0)
            T.ppl_copy(G_right_fp8[bx * block_N + split_N, 0], right_fp8_1)

            T.ppl_copy(right_fp8_0, right_bf16_0)
            T.ppl_copy(right_fp8_1, right_bf16_1)

            scale_n_idx = bx * block_N // group_N
            T.ppl_copy(G_scale_group[scale_n_idx, 0], scale_group_shared)

            T.ppl_npu_bcast(scale_kgroup_0, scale_group_shared)
            T.ppl_mul(right_scaled_0, right_bf16_0, scale_kgroup_0)

            T.ppl_npu_bcast(scale_kgroup_1, scale_group_shared)
            T.ppl_mul(right_scaled_1, right_bf16_1, scale_kgroup_1)

            for m_blk in T.serial(
                T.ceildiv(M, block_M),
                annotations={"tilelang.tpu_core_parallel": True},
            ):
                T.ppl_copy(G_left[m_blk * block_M, 0], left_shared)

                T.ppl_fill(out_shared_0, T.float32(0))
                T.ppl_gemm(left_shared, right_scaled_0, out_shared_0, transpose_B=True)
                T.ppl_copy(out_shared_0, out_cast_0)
                T.ppl_copy(out_cast_0, G_out[m_blk * block_M, bx * block_N])

                T.ppl_fill(out_shared_1, T.float32(0))
                T.ppl_gemm(left_shared, right_scaled_1, out_shared_1, transpose_B=True)
                T.ppl_copy(out_shared_1, out_cast_1)
                T.ppl_copy(out_cast_1, G_out[m_blk * block_M, bx * block_N + split_N])

    return main_kernel_inner


class W8A16ScaleDeviceSplit64A341:
    """A3.4.1 device-side raw-scale path with a model-compatible ABI wrapper."""

    def __init__(
        self,
        M=1024,
        K=128,
        N=128,
        block_M=128,
        block_N=128,
        block_K=128,
        split_N=64,
        group_N=128,
        group_K=128,
    ):
        self.M = M
        self.K = K
        self.N = N
        self.block_M = block_M
        self.block_N = block_N
        self.block_K = block_K
        self.split_N = split_N
        self.group_N = group_N
        self.group_K = group_K

        self.N_GROUP = ceildiv(N, group_N)
        self.K_GROUP = ceildiv(K, group_K)

        self.kernel = tilelang.compile(
            w8a16_mn_mn_right_reuse_bn128_scale_device_split64_A341_kernel(
                M=M,
                K=K,
                N=N,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                split_N=split_N,
                group_N=group_N,
                group_K=group_K,
            ),
            out_idx=[3],
            target="tpu",
            mode="pcie",
            tpu_core_parallel={
                "enable": True,
                "core_num": "auto",
                "max_core_num": 8,
            },
        )

    def check_scale_group(self, scale_group: torch.Tensor):
        expected_shape = (self.N_GROUP, self.K_GROUP)
        if tuple(scale_group.shape) != expected_shape:
            raise ValueError(
                f"scale_group shape mismatch: expected {expected_shape}, "
                f"got {tuple(scale_group.shape)}"
            )

    def __call__(
        self,
        left: torch.Tensor,
        right_fp8: torch.Tensor,
        scale_group: torch.Tensor,
        out: torch.Tensor = None,
    ):
        if tuple(left.shape) != (self.M, self.K):
            raise ValueError(f"left shape mismatch: expected {(self.M, self.K)}, got {tuple(left.shape)}")

        if tuple(right_fp8.shape) != (self.N, self.K):
            raise ValueError(f"right_fp8 shape mismatch: expected {(self.N, self.K)}, got {tuple(right_fp8.shape)}")

        self.check_scale_group(scale_group)

        if out is None:
            out = torch.empty((self.M, self.N), dtype=torch.bfloat16)

        if tuple(out.shape) != (self.M, self.N):
            raise ValueError(f"out shape mismatch: expected {(self.M, self.N)}, got {tuple(out.shape)}")

        ret = self.kernel(left, right_fp8, scale_group, out)
        return ret, out

    def forward_op_abi(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        output: torch.Tensor,
        blocksize: int,
    ):
        """Model/C++ compatible ABI:
        mm_w8a16_dq_forward(input, weight, scale, output, blocksize)

        TileLang device kernel still receives only tensor arguments:
        input, weight, scale, output

        blocksize is checked at wrapper level because this prototype compiles
        group_N/group_K as constants.
        """
        if int(blocksize) != self.group_N or int(blocksize) != self.group_K:
            raise ValueError(
                f"blocksize mismatch: got {blocksize}, "
                f"expected group_N={self.group_N}, group_K={self.group_K}"
            )

        return self(
            left=input,
            right_fp8=weight,
            scale_group=scale,
            out=output,
        )


M, K, N = 8192, 128, 128
block_M = 128
block_N = 128
block_K = 128
split_N = 64
group_N = 128
group_K = 128
blocksize = 128

runner = W8A16ScaleDeviceSplit64A341(
    M=M,
    K=K,
    N=N,
    block_M=block_M,
    block_N=block_N,
    block_K=block_K,
    split_N=split_N,
    group_N=group_N,
    group_K=group_K,
)

print(
    "compile success: W8A16 e4m3 M8192 BN128 right-reuse scale-device split64 A3.4.1 op-abi real auto-core",
    flush=True,
)

torch.manual_seed(0)

left = torch.randn((M, K), dtype=torch.bfloat16)
right_fp8 = torch.randn((N, K), dtype=torch.float32).to(torch.float8_e4m3fn)
right_ref = right_fp8.to(torch.float32)

N_GROUP = ceildiv(N, group_N)
K_GROUP = ceildiv(K, group_K)

scale_group = (torch.rand((N_GROUP, K_GROUP), dtype=torch.float32) * 0.1).to(torch.bfloat16)
scale_group = scale_group.contiguous()

out = torch.empty((M, N), dtype=torch.bfloat16)

# Use the same parameter order as soph_deepseek_v3.py / C++ / PPL:
# input, weight, scale, output, blocksize.
ret, out = runner.forward_op_abi(
    left,
    right_fp8,
    scale_group,
    out,
    blocksize,
)

print("kernel call returned:", ret, flush=True)
print("out finite:", torch.isfinite(out.float()).all().item(), flush=True)

scale_full = expand_scale_group_to_full(
    scale_group=scale_group,
    N=N,
    K=K,
    group_N=group_N,
    group_K=group_K,
)

ref = left.float() @ (right_ref.float() * scale_full.float()).T
ref_bf16 = ref.to(torch.bfloat16)
diff = (out.float() - ref_bf16.float()).abs()

print("scale_group shape:", tuple(scale_group.shape), flush=True)
print("blocksize:", blocksize, flush=True)
print("out max abs:", out.float().abs().max().item(), flush=True)
print("ref max abs:", ref_bf16.float().abs().max().item(), flush=True)
print("diff max:", diff.max().item(), flush=True)
print("diff mean:", diff.double().mean().item(), flush=True)
print(
    "close relaxed:",
    torch.allclose(out.float(), ref_bf16.float(), atol=2e-2, rtol=2e-2),
    flush=True,
)