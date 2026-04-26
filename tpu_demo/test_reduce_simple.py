import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target='tpu', mode='cmodel', out_idx=[1])
def simple_reduce(a: T.Buffer([1, 1, 1, 32], 'float32'), out: T.Buffer([1, 1, 1, 1], 'float32')):
    T.ppl_reduce_sum(a, out, axis=3, keepdims=True)

a_np = torch.ones(1, 1, 1, 32).float().numpy()
out_np = torch.zeros(1, 1, 1, 1).float().numpy()
print("Testing reduce_sum with all ones, expected result: 32.0")
simple_reduce(a_np, out_np)
print(f"Result: {out_np[0,0,0,0]}")
print(f"Check: {'OK' if abs(out_np[0,0,0,0] - 32.0) < 0.1 else 'FAIL'}")
