import os, sys, json, re, torch, torch.nn.functional as F

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCHMARK_ROOT)

import tilelang
from swiglu.kernels import build_tl_swiglu
from rope.kernels import build_tl_rope
from elementwise_add.kernels import build_tl_add
from ppl_utils import compile_ppl_kernel, generate_pl

TORCH_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

def make_rope_cos_sin(C, W, dtype):
    half = W // 2
    k = torch.arange(half, dtype=dtype)
    inv_freq = (10000.0 ** (-2 * k / W)).to(dtype)
    pos = torch.arange(C, dtype=dtype).unsqueeze(1)
    theta = pos * inv_freq
    c, s = torch.cos(theta), torch.sin(theta)
    cos = torch.empty(C, W, dtype=dtype); sin = torch.empty(C, W, dtype=dtype)
    cos[:, 0::2] = c; cos[:, 1::2] = c
    sin[:, 0::2] = s; sin[:, 1::2] = s
    return cos.contiguous(), sin.contiguous()

def make_test_data(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu":
        C, W = shape["C"], shape["W"]
        a, b = torch.randn(C, W, dtype=dt), torch.randn(C, W, dtype=dt)
        ref = (b.float() * F.silu(a.float())).to(dt)
        return [a, b], ref, [2], (C, W)
    elif op == "rope":
        C, W = shape["C"], shape["W"]
        x = torch.randn(C, W, dtype=dt)
        cos, sin = make_rope_cos_sin(C, W, dt)
        out_ref = torch.empty_like(x)
        out_ref[:, 0::2] = x[:, 0::2]*cos[:, 0::2] - x[:, 1::2]*sin[:, 0::2]
        out_ref[:, 1::2] = x[:, 1::2]*cos[:, 1::2] + x[:, 0::2]*sin[:, 1::2]
        return [x, cos, sin], out_ref, [3], (C, W)
    elif op == "add":
        M, N = shape["M"], shape["N"]
        a, b = torch.randn(M, N, dtype=dt), torch.randn(M, N, dtype=dt)
        return [a, b], a + b, [2], (M, N)

def build_tl_func(op, dtype, shape, block):
    if op == "swiglu": return build_tl_swiglu(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "rope": return build_tl_rope(dtype, shape["C"], shape["W"], block["block_C"], block["block_W"])
    elif op == "add":  return build_tl_add(dtype, shape["M"], shape["N"], block["block_M"], block["block_N"])

def build_arg_specs(op, dtype, shape):
    dt = TORCH_DTYPE_MAP[dtype]
    if op == "swiglu": return [((shape["C"], shape["W"]), dt)] * 3
    elif op == "rope": return [((shape["C"], shape["W"]), dt)] * 4
    elif op == "add":  return [((shape["M"], shape["N"]), dt)] * 3

def parse_avg_time(text):
    m = re.search(r"Average execution time:\s+([\d.]+)\s+ms", text)
    return float(m.group(1)) if m else None

cfg = json.loads(sys.argv[1])
op, dtype, shape, block, atol, rtol = cfg["op"], cfg["dtype"], cfg["shape"], cfg["block"], cfg["atol"], cfg["rtol"]
backend = cfg["backend"]  # "tilelang" or "ppl"

dt = TORCH_DTYPE_MAP[dtype]
inputs, ref, result_idx, out_shape = make_test_data(op, dtype, shape)
out = torch.zeros(out_shape, dtype=dt)

if backend == "tilelang":
    tl_func = build_tl_func(op, dtype, shape, block)
    kernel = tilelang.compile(tl_func, out_idx=-1, target="tpu")
    kernel(*inputs, out)
else:
    pl_path = generate_pl(op, dtype, shape)
    arg_specs = build_arg_specs(op, dtype, shape)
    forward = compile_ppl_kernel(pl_path, arg_specs, result_idx=result_idx)
    forward(*inputs, out)

correct = torch.allclose(out, ref, atol=atol, rtol=rtol)
max_diff = (out.float() - ref.float()).abs().max().item()

result = {"correct": correct, "max_diff": max_diff}

import ctypes
try:
    ctypes.CDLL(None).fflush(None)
except Exception:
    pass
sys.stderr.write("RESULT:" + json.dumps(result) + "\n")
sys.stderr.flush()
sys.stdout.flush()
os._exit(0)
