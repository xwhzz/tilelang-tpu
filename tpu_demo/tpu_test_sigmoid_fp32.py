import tilelang
import tilelang.language as T
import torch


def sigmoid_kernel(Block_w, Block_c, C, W, dtype="float32", accum_dtype="float32"):
	global_shape = (C, W)

	@T.prim_func
	def main_kernel_inner(
		G_in: T.Tensor(global_shape, dtype),
		G_out: T.Tensor(global_shape, accum_dtype),
	):
		with T.Kernel(T.ceildiv(C, Block_c), T.ceildiv(W, Block_w), is_cpu=True) as (bx, by):
			block_shape = (Block_c, Block_w)

			x = T.alloc_shared(block_shape, accum_dtype)
			out = T.alloc_shared(block_shape, accum_dtype)

			T.copy(G_in[bx * Block_c, by * Block_w], x)
			T.sigmoid(out, x)
			T.copy(out, G_out[bx * Block_c, by * Block_w])

	return main_kernel_inner


def main():
	C = 1024
	W = 1024
	block_C = 32
	block_W = 32

	kernel = tilelang.compile(
		sigmoid_kernel(Block_c=block_C, Block_w=block_W, C=C, W=W),
		out_idx=-1,
		target="tpu",
	)

	torch.manual_seed(0)
	a = torch.randn(C, W).float()
	out = torch.zeros(C, W).float()

	_ = kernel(a, out)

	ref = torch.sigmoid(a).float()
	diff = ref - out
	abs_diff = diff.abs()
	max_abs_diff = torch.max(abs_diff)
	mean_abs_diff = torch.mean(abs_diff)

	print("output:")
	print(out)
	print("ref:")
	print(ref)
	print("\n=== Difference Stats ===")
	print(f"max abs diff: {max_abs_diff}")
	print(f"mean abs diff: {mean_abs_diff}")
	print("check close (atol=1e-2, rtol=1e-2):")
	print(torch.allclose(out, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
	main()
