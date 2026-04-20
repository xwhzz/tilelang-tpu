# TileLang-TPU

TileLang-TPU is a TPU-oriented extension of TileLang for SOPHGO accelerators. It preserves the TileLang Python DSL while adding TPU lowering, TPU code generation, and JIT runtime integration, enabling TileLang kernels to be compiled and executed on TPU platforms.

The project provides end-to-end TPU support centered on BM1690, including `target="tpu"`, `mode="pcie|cmodel"`, TPU DSL intrinsics, generated host/device glue code, and runnable TPU demos.
## Highlights

- TileLang frontend with TPU target support
- TPU-specific DSL intrinsics such as `ppl_copy`, `ppl_gemm`, `ppl_reduce_sum`, `ppl_reduce_max`, `ppl_rsqrt`, and `ppl_rope_add`
- End-to-end JIT flow from Python kernel definition to generated TPU host/device artifacts
- BM1690-oriented execution path with both PCIe execution and `cmodel` simulation modes
- Operator coverage aligned with kernels commonly used by Llama and DeepSeek workloads

## What This Project Does

TileLang-TPU is a TileLang-to-TPU compiler and runtime path for SOPHGO accelerators.

- It reuses the TileLang frontend and extends it with TPU-specific lowering, codegen, and JIT support.
- It uses the underlying TPU compilation tools also used in the PPL repository, but it is not built as a layer on top of the PPL software stack.
- It focuses on turning TileLang programs into runnable TPU kernels, including host/device wrapper generation and execution flow integration.

In short, this repository is about bringing TileLang's programming model to SOPHGO TPU targets rather than repackaging PPL itself.

## Current Status

- The current mainline JIT path focuses on BM1690.
- `tilelang.compile(..., target="tpu")` is wired into the repository.
- Both `mode="pcie"` and `mode="cmodel"` are present in the TPU adapter path.
- TPU demos are available under [`tpu_demo/`](./tpu_demo/).
- This project is under active development, and pull requests are welcome.

## Requirements

- Linux and Python 3
- SOPHGO TPU toolchain environment
- Access to BM1690 hardware or a working `cmodel` setup

## Key TPU Paths

- [`tilelang/engine/`](./tilelang/engine/)
- [`tilelang/language/`](./tilelang/language/)
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/)
- [`src/target/`](./src/target/)
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/)
- [`src/transform/`](./src/transform/)
- [`tpu_demo/`](./tpu_demo/)

## Quick Start

### 1. Prepare the repository

```bash
git submodule update --init --recursive
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
cd ../..
```

### 2. Build and install

```bash
./install_tpu.sh
pip install -e . -v
```

### 3. Run a TPU demo

```bash
python tpu_demo/matmul/tpu_test_matmul_fp16.py
python tpu_demo/rms_norm/rms_norm_fp16.py
python tpu_demo/tpu_test_swiglu_fp32_cmodel.py
```

For a more detailed setup guide, see [`tpu_demo/instruction.md`](./tpu_demo/instruction.md).

## Programming Model

TileLang-TPU keeps the familiar TileLang workflow and switches the backend to TPU:

```python
import tilelang

kernel = tilelang.compile(
    my_kernel,
    out_idx=-1,
    target="tpu",
    mode="pcie",   # or "cmodel"
)
```

Inside TPU kernels, the common building blocks are exposed as TileLang DSL intrinsics in [`tilelang/language/customize.py`](./tilelang/language/customize.py), including:

- `T.ppl_copy`
- `T.ppl_fill`
- `T.ppl_gemm`
- `T.ppl_reduce_sum`
- `T.ppl_reduce_max`
- `T.ppl_add`, `T.ppl_subtract`, `T.ppl_mul`, `T.ppl_div`
- `T.ppl_add_C`, `T.ppl_mul_C`
- `T.ppl_rsqrt`
- `T.ppl_rope_add`

## Examples

The current examples mainly cover operators commonly used in Llama and DeepSeek workloads, and more operators will be added over time.

Representative examples include:

- Matmul
- RMSNorm
- RoPE
- Reduce
- SwiGLU
- FlashAttention

## Repository Layout

- [`tilelang/`](./tilelang/): TileLang frontend and TPU-facing Python entry points
- [`src/target/`](./src/target/): TPU codegen and runtime modules
- [`src/tl_templates/tpu/`](./src/tl_templates/tpu/): TPU code templates and generated host/device artifacts
- [`tilelang/jit/adapter/`](./tilelang/jit/adapter/): TPU JIT wrapper and library generation flow
- [`tpu_demo/`](./tpu_demo/): TPU demos and bring-up scripts

## Development Notes

- If you modify C++ code, rebuild the native components before rerunning demos:

```bash
make -j 10
```

- Format the repository with:

```bash
./format.sh
```

## Acknowledgements

TileLang-TPU builds on open-source work from [TileLang](https://github.com/tile-ai/tilelang) and uses the TPU compilation ecosystem associated with [SOPHGO PPL](https://github.com/sophgo/PPL/).
