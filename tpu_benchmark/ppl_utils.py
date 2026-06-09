"""
Utility to compile a .pl file via ppl-compile, build it into a shared library
using tilelang's TPU host templates, and return a callable forward function.

Usage:
    from ppl_utils import compile_ppl_kernel
    forward = compile_ppl_kernel("swiglu/pl/swiglu_fp32_64x64.pl", num_args=3, result_idx=[2])
    forward(a, b, c)  # torch tensors
"""

import ctypes
import os
import re
import shutil
import subprocess
import tempfile

import torch


# ── env helpers ───────────────────────────────────────────────────────────────

def get_ppl_project_root():
    root = os.environ.get("PPL_PROJECT_ROOT", None)
    if root is None:
        raise EnvironmentError(
            "PPL_PROJECT_ROOT is not set. "
            "Please export PPL_PROJECT_ROOT=/path/to/ppl_v1.4.195-..."
        )
    return root


def _get_tpu_template_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "../src/tl_templates/tpu"))


# ── Step 1: ppl-compile .pl → rewritten kernel.c ─────────────────────────────

def compile_pl(pl_path, chip="bm1690", opt_level=2):
    pl_path = os.path.abspath(pl_path)
    if not os.path.isfile(pl_path):
        raise FileNotFoundError(f"PL file not found: {pl_path}")

    ppl_root = get_ppl_project_root()
    ppl_compile = os.path.join(ppl_root, "bin", "ppl-compile")

    out_dir = tempfile.mkdtemp(prefix="ppl_build_")
    cmd = [
        ppl_compile, pl_path, "--function=*",
        "-chip", chip, f"-O{opt_level}", "-o", out_dir,
    ]
    env = os.environ.copy()
    env["PPL_PROJECT_ROOT"] = ppl_root
    env["PATH"] = os.path.join(ppl_root, "bin") + ":" + env.get("PATH", "")

    ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ppl-compile failed:\n{ret.stdout}\n{ret.stderr}")

    basename = os.path.splitext(os.path.basename(pl_path))[0]
    device_c = os.path.join(out_dir, "device", basename + ".c")
    if not os.path.isfile(device_c):
        device_dir = os.path.join(out_dir, "device")
        candidates = os.listdir(device_dir) if os.path.isdir(device_dir) else []
        raise FileNotFoundError(
            f"Expected {device_c}, found: {candidates}")

    return _rewrite_kernel_c(device_c)


def _rewrite_kernel_c(device_c_path):
    with open(device_c_path, "r") as f:
        src = f.read()

    m = re.search(r"TPUKERNEL_FUNC_REGISTER\((\w+)\)", src)
    if not m:
        raise ValueError("Cannot find TPUKERNEL_FUNC_REGISTER in generated kernel.c")
    orig_name = m.group(1)

    if orig_name != "main_kernel":
        src = src.replace(orig_name + "_inner", "main_kernel_inner")
        src = src.replace(f"tpu_kernel_api_{orig_name}_t",
                          "tpu_kernel_api_main_inner_args_t")
        src = src.replace(f"TPUKERNEL_FUNC_REGISTER({orig_name})",
                          "TPUKERNEL_FUNC_REGISTER(main_kernel)")
        src = src.replace(f"int {orig_name}(const void",
                          "int main_kernel(const void")
        src = src.replace(f"_tensor__{orig_name}_t", "__ppl_tensor_info")
    else:
        src = src.replace("tpu_kernel_api_main_kernel_t",
                          "tpu_kernel_api_main_inner_args_t")
        src = src.replace("_tensor__main_kernel_t", "__ppl_tensor_info")

    out_path = os.path.join(os.path.dirname(device_c_path), "kernel_rewritten.c")
    with open(out_path, "w") as f:
        f.write(src)
    return out_path


# ── Step 2: generate host template files ──────────────────────────────────────

def _dtype_sizeof(ppl_dtype):
    _MAP = {
        "DT_FP32": 4, "DT_INT32": 4,
        "DT_FP16": 2, "DT_BFP16": 2, "DT_INT16": 2,
        "DT_INT8": 1, "DT_UINT8": 1,
    }
    return _MAP.get(ppl_dtype, 4)


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _generate_templates(build_dir, num_args, dtype_map, shapes, result_idx):
    """Generate kernel.h, kernel.cpp, main.cpp into build_dir.

    Args:
        build_dir: where to write the generated files
        num_args: number of tensor arguments
        dtype_map: list of PPL dtype strings, e.g. ["DT_FP32", "DT_FP32", "DT_FP32"]
        shapes: list of shape tuples (arbitrary ndim) for each arg
        result_idx: list of output arg indices (for D2S copy)
    """
    tpl_dir = _get_tpu_template_dir()
    param_names = [f"ptr_v{i+1}" for i in range(num_args)]

    # kernel.h
    with open(os.path.join(tpl_dir, "kernel_template.h")) as f:
        h_tpl = f.read()
    h_content = h_tpl.format(
        struct_members="\n  ".join(f"unsigned long long {n};" for n in param_names),
        function_name="main_kernel",
        func_params=", ".join(f"unsigned long long {n}" for n in param_names),
        struct_params=", ".join(f"unsigned long long {n}" for n in param_names),
    )
    with open(os.path.join(build_dir, "kernel.h"), "w") as f:
        f.write(h_content)

    # kernel.cpp
    with open(os.path.join(tpl_dir, "kernel_template.cpp")) as f:
        cpp_tpl = f.read()
    cpp_content = cpp_tpl.format(
        function_name="main_kernel",
        func_params=", ".join(f"unsigned long long {n}" for n in param_names),
        struct_assignments="\n  ".join(f"api.{n} = {n};" for n in param_names),
    )
    with open(os.path.join(build_dir, "kernel.cpp"), "w") as f:
        f.write(cpp_content)

    # main.cpp
    with open(os.path.join(tpl_dir, "main_template.cpp")) as f:
        main_tpl = f.read()

    arg_decls, dev_decls, mallocs, s2d, d2s, frees, call_args = [], [], [], [], [], [], []
    for i in range(num_args):
        name = f"arg{i}"
        nbytes = _numel(shapes[i]) * _dtype_sizeof(dtype_map[i])
        arg_decls.append(f"  char* {name} = (char*)args[{i}];")
        arg_decls.append(f"  size_t {name}_size = {nbytes};")
        dev_decls.append(f"  void *dev_{name};")
        mallocs.append(f"  tpuRtMalloc((void **)(&dev_{name}), {name}_size, 0);")
        s2d.append(f"  tpuRtMemcpyS2D(dev_{name}, {name}, {name}_size);")
        if i in result_idx:
            d2s.append(f"  tpuRtMemcpyD2S({name}, dev_{name}, {name}_size);")
        frees.append(f"  tpuRtFree(&dev_{name}, 0);")
        call_args.append(f"(unsigned long long)dev_{name}")

    kernel_call = f'  int rst = main_kernel({", ".join(call_args)});'
    pure_kernel_call = f'  rst = main_kernel({", ".join(call_args)});'

    main_content = main_tpl.format(
        arg_declarations="\n".join(arg_decls),
        device_declarations="\n".join(dev_decls),
        malloc_statements="\n".join(mallocs),
        memcpy_s2d_statements="\n".join(s2d),
        memcpy_d2s_statements="\n".join(d2s),
        free_statements="\n".join(frees),
        kernel_call=kernel_call,
        pure_kernel_call=pure_kernel_call,
    )
    with open(os.path.join(build_dir, "main.cpp"), "w") as f:
        f.write(main_content)


# ── Step 3: compile everything → main.so ──────────────────────────────────────

def _compile_to_so(build_dir, kernel_c_path, mode="pcie"):
    PPL_TOP = get_ppl_project_root()
    CHIP = "bm1690"

    includes = [
        # "-I/lib/x86_64-linux-gnu/",
        f"-I{build_dir}",
        f"-I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include",
        f"-I{PPL_TOP}/runtime/kernel",
        f"-I{PPL_TOP}/runtime/customize/include",
        f"-I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include",
    ]
    lib_paths = [
        # "-L/lib/x86_64-linux-gnu/",
        f"-L{PPL_TOP}/runtime/{CHIP}/lib",
        "-L/opt/tpuv7/tpuv7-current/lib/",
        f"-L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib",
    ]
    rpath = f"-Wl,-rpath,{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib"

    TOOLCHAIN = f"{PPL_TOP}/third_party/toolchains_dir/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1"
    CROSS_GCC = f"{TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")

    # Copy kernel.c into build_dir
    shutil.copy2(kernel_c_path, os.path.join(build_dir, "kernel.c"))

    # 1) cross-compile kernel.c → kernel.o
    run([CROSS_GCC, "-D__bm1690__", "-Dlibkernel_EXPORTS", *includes,
         "-Wl,--no-undefined", "-fPIC", "-c",
         os.path.join(build_dir, "kernel.c"),
         "-o", os.path.join(build_dir, "kernel.o")])

    # 2) cross-compile ppl_helper.c → ppl_helper.o
    run([CROSS_GCC, "-D__bm1690__", "-Dlibkernel_EXPORTS", *includes,
         "-Wl,--no-undefined", "-fPIC", "-c",
         f"{PPL_TOP}/runtime/customize/src/ppl_helper.c",
         "-o", os.path.join(build_dir, "ppl_helper.o")])

    # 3) link → libkernel.so
    run([CROSS_GCC, "-fPIC", "-Wl,--no-undefined", "-shared",
         "-Wl,-soname,libkernel.so",
         "-o", os.path.join(build_dir, "libkernel.so"),
         os.path.join(build_dir, "kernel.o"),
         os.path.join(build_dir, "ppl_helper.o"),
         *lib_paths, rpath,
         "-Wl,--whole-archive", "-Wl,-Bstatic", f"-l{CHIP}",
         "-Wl,-Bdynamic", "-Wl,--no-whole-archive", "-lm"])

    # 4) compile kernel.cpp → kernel_host.o
    run(["g++", f"-D__{CHIP}__", *includes, "-Wl,--no-undefined",
         "-std=c++11", "-fPIC", "-c",
         os.path.join(build_dir, "kernel.cpp"),
         "-o", os.path.join(build_dir, "kernel_host.o")])

    # 5) compile main.cpp → main.o
    run(["g++", f"-D__{CHIP}__", *includes, "-Wl,--no-undefined",
         "-std=c++11", "-fPIC", "-c",
         os.path.join(build_dir, "main.cpp"),
         "-o", os.path.join(build_dir, "main.o")])

    # 6) link → main.so
    main_so = os.path.join(build_dir, "main.so")
    run(["g++", "-shared", "-fPIC", "-Wl,--no-undefined",
         "-o", main_so,
         os.path.join(build_dir, "kernel_host.o"),
         os.path.join(build_dir, "main.o"),
         *lib_paths, rpath,
         "-ltpuv7_rt", "-lcdm_daemon_emulator", "-lpthread"])

    return main_so


# ── Step 4: wrap into a callable ──────────────────────────────────────────────

def _make_forward(main_so_path, result_idx):
    lib = ctypes.CDLL(main_so_path)
    lib.tilelang_tpu_run.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.tilelang_tpu_run.restype = ctypes.c_int

    def forward(*args):
        host_tensors = [arg.detach().cpu().contiguous() for arg in args]

        argv = (ctypes.c_void_p * len(args))()
        for i, tensor in enumerate(host_tensors):
            argv[i] = ctypes.c_void_p(tensor.data_ptr()).value

        ret = lib.tilelang_tpu_run(argv)
        for i in result_idx:
            args[i][...] = host_tensors[i].to(args[i].device)
        return ret

    return forward


# ── public API ────────────────────────────────────────────────────────────────

_TORCH_TO_PPL_DTYPE = {
    torch.float32: "DT_FP32",
    torch.float16: "DT_FP16",
    torch.bfloat16: "DT_BFP16",
    torch.int32: "DT_INT32",
}


def compile_ppl_kernel(pl_path, arg_specs, result_idx):
    """Compile a .pl into a callable kernel via tilelang's host framework.

    Args:
        pl_path: path to the .pl file
        arg_specs: list of (shape_tuple, torch_dtype) for each kernel argument,
                   e.g. [((64,64), torch.float32), ((64,64), torch.float32), ...]
        result_idx: list of output argument indices (0-based)

    Returns:
        A callable: forward(tensor_0, tensor_1, ...) -> int
    """
    kernel_c = compile_pl(pl_path)

    build_dir = tempfile.mkdtemp(prefix="ppl_tl_build_")

    num_args = len(arg_specs)
    dtype_map = [_TORCH_TO_PPL_DTYPE[spec[1]] for spec in arg_specs]
    shapes = [spec[0] for spec in arg_specs]

    # Set PPL_KERNEL_PATH for the runtime to find libkernel.so
    os.environ["PPL_KERNEL_PATH"] = os.path.join(build_dir, "libkernel.so")

    _generate_templates(build_dir, num_args, dtype_map, shapes, result_idx)
    main_so = _compile_to_so(build_dir, kernel_c)
    forward = _make_forward(main_so, result_idx)

    return forward


# ── .pl template generation ───────────────────────────────────────────────────

_PPL_TYPE = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}

_SWIGLU_TEMPLATE = '''\
#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_in, {ppl_t} *ptr_right, {ppl_t} *ptr_out) {{
  int C = {C};
  int W = {W};
  int block_w = 16;
  int block_c = 64;

  dim4 global_shape = {{1, C, 1, W}};
  auto g_in = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_in);
  auto g_right = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_right);
  auto g_out = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_out);

  dim4 block_shape = {{1, block_c, 1, block_w}};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {{
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {{
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);

      dim4 in_shape = {{1, c, 1, w}};
      dim4 in_offset = {{0, idx_c, 0, idx_w}};
{load_cast_block}
      auto x_neg = make_tensor<fp32>(block_shape, in_shape);
      tiu::fmul(x_neg, in, -1.0);
      auto x_neg_exp = make_tensor<fp32>(block_shape, in_shape);
      exp_no_overflow(x_neg_exp, x_neg, &block_shape, &in_shape);
      auto x_neg_exp_1 = make_tensor<fp32>(block_shape, in_shape);
      tiu::fadd(x_neg_exp_1, x_neg_exp, 1.0);
      auto x_neg_exp_1_div = make_tensor<fp32>(block_shape, in_shape);
      tiu::fdiv(x_neg_exp_1_div, in, x_neg_exp_1, 3);
      auto out_fp32 = make_tensor<fp32>(block_shape, in_shape);
      tiu::fmul(out_fp32, x_neg_exp_1_div, right);
{store_cast_block}
    }}
  }}
}}
'''

_ROPE_TEMPLATE = '''\
#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_in, {ppl_t} *ptr_cos, {ppl_t} *ptr_sin,
                            {ppl_t} *ptr_out) {{
  int C = {C};
  int W = {W};
  int block_w = 16;
  int block_c = 64;

  dim4 global_shape = {{1, C, 1, W}};
  auto g_in = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_in);
  auto g_out = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_out);
  auto g_cos = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_cos);
  auto g_sin = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_sin);

  dim4 block_shape = {{1, block_c, 1, block_w}};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {{
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {{
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);

      dim4 in_shape = {{1, c, 1, w}};
      auto in = make_tensor<{ppl_t}>(block_shape, in_shape);
      auto in_cos = make_tensor<{ppl_t}>(block_shape, in_shape);
      auto in_sin = make_tensor<{ppl_t}>(block_shape, in_shape);

      dim4 in_offset = {{0, idx_c, 0, idx_w}};
      dma::load(in, g_in.sub_view(in_shape, in_offset));
      dma::load(in_cos, g_cos.sub_view(in_shape, in_offset));
      dma::load(in_sin, g_sin.sub_view(in_shape, in_offset));

      auto x_cos = make_tensor<{ppl_t}>(block_shape, in_shape);
      tiu::fmul(x_cos, in, in_cos);
      auto x_sin = make_tensor<{ppl_t}>(block_shape, in_shape);
      tiu::fmul(x_sin, in, in_sin);
      auto x_neg = make_tensor<{ppl_t}>(block_shape, in_shape);
      tiu::fmul(x_neg, in, -1.0);
      auto x_neg_sin = make_tensor<{ppl_t}>(block_shape, in_shape);
      tiu::fmul(x_neg_sin, x_neg, in_sin);

      auto out = make_tensor<{ppl_t}>(block_shape, in_shape);
      dim4 half_shape = {{1, c, 1, w / 2}};
      dim4 half_stride;
      get_stride(&half_stride, &in_shape, TPU_ALIGN, get_eu_num<{ppl_t}>());
      half_stride.w = 2;
      dim4 offset = {{0, 0, 0, 1}};
      auto x_neg_sin_half =
          x_neg_sin.sub_view(half_shape, offset).view(half_shape, half_stride);
      tiu::fadd(out.view(half_shape, half_stride),
                x_cos.view(half_shape, half_stride), x_neg_sin_half);
      tiu::fadd(
          out.sub_view(half_shape, offset).view(half_shape, half_stride),
          x_cos.sub_view(half_shape, offset).view(half_shape, half_stride),
          x_sin.view(half_shape, half_stride));
      dma::store(g_out.sub_view(in_shape, in_offset), out);
    }}
  }}
}}
'''

_ADD_TEMPLATE = '''\
#include "ppl.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_a, {ppl_t} *ptr_b, {ppl_t} *ptr_c) {{
  int C = {M};
  int W = {N};
  int block_c = 64;
  int block_w = 16;

  dim4 global_shape = {{1, C, 1, W}};
  auto g_a = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_a);
  auto g_b = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_b);
  auto g_c = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_c);

  dim4 block_shape = {{1, block_c, 1, block_w}};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {{
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {{
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);
      dim4 in_shape = {{1, c, 1, w}};
      auto a = make_tensor<{ppl_t}>(block_shape, in_shape);
      auto b = make_tensor<{ppl_t}>(block_shape, in_shape);
      auto out = make_tensor<{ppl_t}>(block_shape, in_shape);
      dim4 in_offset = {{0, idx_c, 0, idx_w}};
      dma::load(a, g_a.sub_view(in_shape, in_offset));
      dma::load(b, g_b.sub_view(in_shape, in_offset));
      tiu::fadd(out, a, b);
      dma::store(g_c.sub_view(in_shape, in_offset), out);
    }}
  }}
}}
'''


_MATMUL_TEMPLATE = '''\
#include "ppl.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_a, {ppl_t} *ptr_b, {ppl_t} *ptr_c) {{
  int M = {M};
  int K = {K};
  int N = {N};
  int block_m = {block_M};
  int block_k = {block_K};
  int block_n = {block_N};

  dim4 a_global_shape = {{1, M, 1, K}};
  dim4 b_global_shape = {{1, K, 1, N}};
  dim4 c_global_shape = {{1, M, 1, N}};
  auto g_a = gtensor<{ppl_t}>(a_global_shape, GLOBAL, ptr_a);
  auto g_b = gtensor<{ppl_t}>(b_global_shape, GLOBAL, ptr_b);
  auto g_c = gtensor<{ppl_t}>(c_global_shape, GLOBAL, ptr_c);

  dim4 a_block_shape = {{1, block_m, 1, block_k}};
  dim4 b_block_shape = {{1, block_k, 1, block_n}};
  dim4 c_block_shape = {{1, block_m, 1, block_n}};

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {{
    int m = min(block_m, M - idx_m);
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {{
      int n = min(block_n, N - idx_n);
      dim4 c_shape = {{1, m, 1, n}};
      dim4 c_offset = {{0, idx_m, 0, idx_n}};
      auto c_local = make_tensor<fp32>(c_block_shape, c_shape);
      tiu::fill(c_local, 0.0f);

      for (int idx_k = 0; idx_k < K; idx_k += block_k) {{
        ppl::enable_pipeline();
        int k = min(block_k, K - idx_k);
        dim4 a_shape = {{1, m, 1, k}};
        dim4 a_offset = {{0, idx_m, 0, idx_k}};
        dim4 b_shape = {{1, k, 1, n}};
        dim4 b_offset = {{0, idx_k, 0, idx_n}};
{load_cast_a}
{load_cast_b}
        tiu::fmm(c_local, a_fp16, b_fp16, true);
      }}
{store_cast_c}
    }}
  }}
}}
'''

_RMSNORM_TEMPLATE = '''\
#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_input, {ppl_t} *ptr_output) {{
  int M = {M};
  int N = {N};
  int block_m = {block_M};
  float eps = 1e-12f;

  dim4 global_shape = {{1, M, 1, N}};
  auto g_in = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_input);
  auto g_out = gtensor<{ppl_t}>(global_shape, GLOBAL, ptr_output);

  dim4 block_shape = {{1, block_m, 1, N}};
  dim4 avg_block_shape = {{1, block_m, 1, 1}};

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {{
    int m = min(block_m, M - idx_m);
    dim4 in_shape = {{1, m, 1, N}};
    dim4 in_offset = {{0, idx_m, 0, 0}};
    dim4 avg_shape = {{1, m, 1, 1}};

{load_block}

    auto in_sq = make_tensor<fp32>(block_shape, in_shape);
    tiu::fmul(in_sq, in_fp32, in_fp32);
    auto avg_buf = make_tensor<fp32>(avg_block_shape, avg_shape);
    quick_pooling(avg_buf, in_sq, &block_shape, &in_shape, 0, 1, 1.0f / N);
    tiu::fadd(avg_buf, avg_buf, eps);

    auto mu = make_tensor<fp32>(avg_block_shape, avg_shape);
    tiu::frsqrt(mu, avg_buf, 4);
    tiu::fmul(in_fp32, in_fp32, mu);

{store_block}
  }}
}}
'''

_FLASHATTN_TEMPLATE = '''\
#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel({ppl_t} *ptr_q, {ppl_t} *ptr_k, {ppl_t} *ptr_v,
                            {ppl_t} *ptr_out) {{
  int batch = {batch};
  int heads = {heads};
  int seq_len = {seq_len};
  int dim = {dim};
  float sqrt_d = 1.0f / sqrtf((float)dim);
  int block_m = {block_M};
  int block_k = {block_N};
  int block_h = heads;

  int order[4] = {{2, 1, 0, 3}};
  dim4 qo_global_shape = {{batch, seq_len, heads, dim}};
  auto q_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_q, order);
  auto k_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_k, order);
  auto v_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_v, order);
  auto out_global = make_gtensor_permute<{ppl_t}>(qo_global_shape, GLOBAL, ptr_out, order);

  dim4 qk_shape = {{block_h, block_m, 1, block_k}};
  dim4 mi_shape = {{block_h, block_m, 1, 1}};
  dim4 acc_shape = {{block_h, block_m, 1, dim}};

  for (int _b = 0; _b < batch; _b++) {{
    for (int _h = 0; _h < heads; _h += block_h) {{
      int real_h = min(block_h, heads - _h);
      for (int _m = 0; _m < seq_len; _m += block_m) {{
        int real_m = min(block_m, seq_len - _m);
        dim4 qi_real_shape = {{real_h, real_m, 1, dim}};
        dim4 qi_offset = {{_h, _m, _b, 0}};
        tensor<{ppl_t}> qi_tensor;
        dma::load(qi_tensor, q_global.sub_view(qi_real_shape, qi_offset));
        tensor<{ppl_t}> qi_scaled;
        tiu::fmul(qi_scaled, qi_tensor, sqrt_d);

        dim4 mi_real = {{real_h, real_m, 1, 1}};
        dim4 acc_real = {{real_h, real_m, 1, dim}};
        auto mi_sub = make_tensor<{ppl_t}>(mi_shape, mi_real);
        auto li_sub = make_tensor<{ppl_t}>(mi_shape, mi_real);
        auto acc_sub = make_tensor<{ppl_t}>(acc_shape, acc_real);
        tiu::fill(mi_sub, -15000);
        tiu::zero(li_sub);
        tiu::zero(acc_sub);

        int block_k_iter = max(min(block_k, seq_len / 2), 1);
        for (int _k = 0; _k < seq_len; _k += block_k_iter) {{
          ppl::enable_pipeline();
          int real_k = min(block_k_iter, seq_len - _k);
          dim4 kvi_real = {{real_h, real_k, 1, dim}};
          dim4 kvi_offset = {{_h, _k, _b, 0}};
          dim4 qk_real = {{real_h, real_m, 1, real_k}};

          tensor<{ppl_t}> ki_tensor, vi_tensor;
          dma::load(ki_tensor, k_global.sub_view(kvi_real, kvi_offset));
          dma::load(vi_tensor, v_global.sub_view(kvi_real, kvi_offset));

          dim4 qk_batch = {{1, real_m, 1, real_k}};
          dim4 qi_batch = {{1, real_m, 1, dim}};
          dim4 ki_batch = {{1, real_k, 1, dim}};
          auto qk_sub = make_tensor<{ppl_t}>(qk_shape, qk_real);
          auto ki_sub = ki_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {{
            dim4 bo = {{i, 0, 0, 0}};
            tiu::fmm2(qk_sub.sub_view(qk_batch, bo),
                      qi_scaled.sub_view(qi_batch, bo),
                      ki_sub.sub_view(ki_batch, bo), false, true, false);
          }}

          tensor<{ppl_t}> max_out, mi_new;
          quick_pooling(max_out, qk_sub, &qk_shape, &qk_real, -15000, 0);
          tiu::fmax(mi_new, mi_sub, max_out);

          tensor<{ppl_t}> alpha, sub_out, li_tmp;
          tiu::fsub(sub_out, mi_sub, mi_new);
          tiu::move(mi_sub, mi_new);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real);
          tiu::fmul(acc_sub, acc_sub, alpha);
          tiu::fmul(li_tmp, li_sub, alpha);

          tensor<{ppl_t}> sub_out1, p_T, sum_val;
          tiu::fsub(sub_out1, qk_sub, mi_new);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real);
          quick_pooling(sum_val, p_T, &qk_shape, &qk_real, 0, 1);
          tiu::fadd(li_sub, li_tmp, sum_val);

          auto pv = make_tensor<{ppl_t}>(acc_shape, acc_real);
          dim4 pv_batch = {{1, real_m, 1, dim}};
          dim4 p_batch = {{1, real_m, 1, real_k}};
          dim4 vi_batch = {{1, real_k, 1, dim}};
          auto vi_sub = vi_tensor.view(kvi_real);
          for (int i = 0; i < real_h; i++) {{
            dim4 bo = {{i, 0, 0, 0}};
            tiu::fmm2(pv.sub_view(pv_batch, bo),
                      p_T.sub_view(p_batch, bo),
                      vi_sub.sub_view(vi_batch, bo));
          }}
          tiu::fadd(acc_sub, acc_sub, pv);
        }}

        tensor<fp32> li_fp32, div_li;
        tensor<{ppl_t}> div_li_T, final_out;
        tiu::cast(li_fp32, li_sub);
        tiu::fdiv(div_li, 1.0f, li_fp32, 3);
        tiu::cast(div_li_T, div_li);
        tiu::fmul(final_out, acc_sub, div_li_T);
        dma::store(out_global.sub_view(acc_real, qi_offset), final_out);
      }}
    }}
  }}
}}
'''


def generate_pl(op, dtype, shape):
    """Generate a .pl file for the given op/dtype/shape and return its path."""
    ppl_t = _PPL_TYPE[dtype]

    if op == "swiglu":
        C, W = shape["C"], shape["W"]
        if dtype == "float32":
            load_cast = (
                "      auto in = make_tensor<fp32>(block_shape, in_shape);\n"
                "      auto right = make_tensor<fp32>(block_shape, in_shape);\n"
                "      dma::load(in, g_in.sub_view(in_shape, in_offset));\n"
                "      dma::load(right, g_right.sub_view(in_shape, in_offset));\n"
            )
            store_cast = (
                "      dma::store(g_out.sub_view(in_shape, in_offset), out_fp32);\n"
            )
        else:
            load_cast = (
                f"      auto in_{ppl_t} = make_tensor<{ppl_t}>(block_shape, in_shape);\n"
                f"      auto right_{ppl_t} = make_tensor<{ppl_t}>(block_shape, in_shape);\n"
                f"      dma::load(in_{ppl_t}, g_in.sub_view(in_shape, in_offset));\n"
                f"      dma::load(right_{ppl_t}, g_right.sub_view(in_shape, in_offset));\n"
                f"      auto in = make_tensor<fp32>(block_shape, in_shape);\n"
                f"      auto right = make_tensor<fp32>(block_shape, in_shape);\n"
                f"      tiu::cast(in, in_{ppl_t});\n"
                f"      tiu::cast(right, right_{ppl_t});\n"
            )
            store_cast = (
                f"      auto out_{ppl_t} = make_tensor<{ppl_t}>(block_shape, in_shape);\n"
                f"      tiu::cast(out_{ppl_t}, out_fp32);\n"
                f"      dma::store(g_out.sub_view(in_shape, in_offset), out_{ppl_t});\n"
            )
        src = _SWIGLU_TEMPLATE.format(
            ppl_t=ppl_t, C=C, W=W,
            load_cast_block=load_cast, store_cast_block=store_cast,
        )
    elif op == "rope":
        src = _ROPE_TEMPLATE.format(ppl_t=ppl_t, C=shape["C"], W=shape["W"])
    elif op == "add":
        src = _ADD_TEMPLATE.format(ppl_t=ppl_t, M=shape["M"], N=shape["N"])
    elif op == "matmul":
        M, K, N = shape["M"], shape["K"], shape["N"]
        block_M = min(32, M)
        block_K = min(32, K)
        block_N = min(32, N)
        if dtype == "float32":
            load_a = (
                "        auto a_local = make_tensor<fp32>(a_block_shape, a_shape);\n"
                "        dma::load(a_local, g_a.sub_view(a_shape, a_offset));\n"
                "        auto a_fp16 = make_tensor<fp16>(a_block_shape, a_shape);\n"
                "        tiu::cast(a_fp16, a_local);"
            )
            load_b = (
                "        auto b_local = make_tensor<fp32>(b_block_shape, b_shape);\n"
                "        dma::load(b_local, g_b.sub_view(b_shape, b_offset));\n"
                "        auto b_fp16 = make_tensor<fp16>(b_block_shape, b_shape);\n"
                "        tiu::cast(b_fp16, b_local);"
            )
            store_c = (
                "      auto c_out = make_tensor<fp32>(c_block_shape, c_shape);\n"
                "      tiu::move(c_out, c_local);\n"
                "      dma::store(g_c.sub_view(c_shape, c_offset), c_out);"
            )
        else:
            load_a = (
                f"        auto a_local = make_tensor<{ppl_t}>(a_block_shape, a_shape);\n"
                f"        dma::load(a_local, g_a.sub_view(a_shape, a_offset));\n"
                f"        auto a_fp16 = a_local;"
            )
            load_b = (
                f"        auto b_local = make_tensor<{ppl_t}>(b_block_shape, b_shape);\n"
                f"        dma::load(b_local, g_b.sub_view(b_shape, b_offset));\n"
                f"        auto b_fp16 = b_local;"
            )
            store_c = (
                f"      auto c_cast = make_tensor<{ppl_t}>(c_block_shape, c_shape);\n"
                f"      tiu::cast(c_cast, c_local);\n"
                f"      dma::store(g_c.sub_view(c_shape, c_offset), c_cast);"
            )
        src = _MATMUL_TEMPLATE.format(
            ppl_t=ppl_t, M=M, K=K, N=N,
            block_M=block_M, block_K=block_K, block_N=block_N,
            load_cast_a=load_a, load_cast_b=load_b, store_cast_c=store_c,
        )
    elif op == "rmsnorm":
        M, N = shape["M"], shape["N"]
        block_M = min(32, M)
        if dtype == "float32":
            load_block = (
                "    auto in_local = make_tensor<fp32>(block_shape, in_shape);\n"
                "    dma::load(in_local, g_in.sub_view(in_shape, in_offset));\n"
                "    auto in_fp32 = in_local;"
            )
            store_block = (
                "    dma::store(g_out.sub_view(in_shape, in_offset), in_fp32);"
            )
        else:
            load_block = (
                f"    auto in_local = make_tensor<{ppl_t}>(block_shape, in_shape);\n"
                f"    dma::load(in_local, g_in.sub_view(in_shape, in_offset));\n"
                f"    auto in_fp32 = make_tensor<fp32>(block_shape, in_shape);\n"
                f"    tiu::cast(in_fp32, in_local);"
            )
            store_block = (
                f"    auto out_cast = make_tensor<{ppl_t}>(block_shape, in_shape);\n"
                f"    tiu::cast(out_cast, in_fp32);\n"
                f"    dma::store(g_out.sub_view(in_shape, in_offset), out_cast);"
            )
        src = _RMSNORM_TEMPLATE.format(
            ppl_t=ppl_t, M=M, N=N, block_M=block_M,
            load_block=load_block, store_block=store_block,
        )
    elif op == "flashattn":
        src = _FLASHATTN_TEMPLATE.format(
            ppl_t=ppl_t,
            batch=shape["batch"], heads=shape["heads"],
            seq_len=shape["seq_len"], dim=shape["dim"],
            block_M=shape.get("block_M", 2), block_N=shape.get("block_N", 2),
        )
    else:
        raise ValueError(f"Unknown op: {op}")

    out_dir = tempfile.mkdtemp(prefix="ppl_pl_")
    ds = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}[dtype]
    if op in ("swiglu", "rope"):
        fname = f"{op}_{ds}_{shape['C']}x{shape['W']}.pl"
    elif op == "matmul":
        fname = f"{op}_{ds}_{shape['M']}x{shape['K']}x{shape['N']}.pl"
    elif op == "rmsnorm":
        fname = f"{op}_{ds}_{shape['M']}x{shape['N']}.pl"
    elif op == "flashattn":
        fname = f"{op}_{ds}_b{shape['batch']}_h{shape['heads']}_s{shape['seq_len']}_d{shape['dim']}.pl"
    else:
        fname = f"{op}_{ds}.pl"
    pl_path = os.path.join(out_dir, fname)
    with open(pl_path, "w") as f:
        f.write(src)
    return pl_path
