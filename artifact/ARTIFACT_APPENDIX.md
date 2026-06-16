# Artifact Appendix

This appendix keeps the detailed artifact information out of the main README while staying inside the submitted artifact package.

## Submission Metadata

### Abstract

This artifact contains the TileLang-TPU compiler/runtime implementation and benchmark suite used by the paper "Practical: Retargeting AI Kernel DSLs Beyond GPUs: An Experience Report on Refactoring TileLang to Sophgo TPUs". It supports BM1690 PCIe performance reproduction for the paper's standalone operator results, fused FlashAttention and MLP W8A16 results, and pipeline/address-allocation ablation. It also provides cmodel simulator checks for reviewers without a physical BM1690 card; cmodel validates correctness only and must not be used for performance claims.

### Badge Request

- Primary: Artifacts Evaluated, Functional.
- Secondary: Artifacts Evaluated, Reusable, if the reviewer has access to SOPHGO SDK/PPL and can run new configurations.
- Available badge: defer until a public DOI archive exists after acceptance.

### Hardware Dependencies

Performance reproduction requires a SOPHGO BM1690 PCIe TPU card, Ubuntu 24.04 LTS, SOPHGO driver v1.9.1, matching firmware, and SOPHGO PPL SDK v1.4.195. Cmodel correctness checks require the SOPHGO emulator runtime but no physical TPU card.

### Claims Covered

| Paper result | Artifact command |
|---|---|
| Figure 4 standalone operator speedups | `python artifact/scripts/run_core.py --figure fig4` |
| Figure 5 FlashAttention and MLP W8A16 | `python artifact/scripts/run_core.py --figure fig5` |
| Figure 6 pipeline/address ablation | `python artifact/scripts/run_ablation.py` |
| Correctness-only simulator check | `python artifact/scripts/run_smoke.py --mode cmodel` |

### Expected Runtime

| Path | Expected time |
|---|---:|
| Cmodel smoke | Minutes to tens of minutes, depending on emulator speed. |
| PCIe smoke | 10 to 30 minutes. |
| Figure 4 | Hours on a BM1690 server. |
| Figure 5 | Hours on a BM1690 server. |
| Figure 6 | Hours on a BM1690 server. |
| Full reproduction | Overnight. |

## Detailed Installation

Run commands from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-build.txt
pip install -r requirements.txt
pip install -r requirements-artifact.txt
```

Apply the TVM patch before building TileLang-TPU:

```bash
git submodule update --init --recursive
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
cd ../..
```

Build and install TileLang-TPU:

```bash
./install_tpu.sh
pip install -e . -v
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Configure the SOPHGO/PPL runtime. Replace `/path/to/ppl` with the installed PPL SDK directory:

```bash
export PPL_PROJECT_ROOT=/path/to/ppl
export SRC_DIR="$PWD/src/tl_templates/tpu"
export TPU_KERNEL_PATH="$SRC_DIR"
export PPL_KERNEL_PATH="$SRC_DIR/libkernel.so"
export LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib:${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator/lib:${LD_LIBRARY_PATH}"
```

If you modify C++ sources under `src/`, rebuild native components before rerunning:

```bash
make -C build -j10
```

## Environment Checks

```bash
python artifact/scripts/check_env.py --mode pcie
python artifact/scripts/check_env.py --mode cmodel
```

The checker reports missing files, missing Python imports, missing environment variables, and the current git commit. It does not modify the repository.

## Detailed Run Notes

### Smoke Suites

PCIe smoke checks TileLang-TPU JIT, PPL baseline calls, correctness comparison, output parsing, and manifest generation:

```bash
python artifact/scripts/run_smoke.py --mode pcie
```

Cmodel smoke uses a small SwiGLU example in simulator mode:

```bash
python artifact/scripts/run_smoke.py --mode cmodel
```

Use `--dry-run` to print commands without executing benchmarks:

```bash
python artifact/scripts/run_smoke.py --mode pcie --dry-run
python artifact/scripts/run_core.py --figure fig4 --dry-run
```

### Figure 4: Standalone Operators

Figure 4 covers Add, RoPE, SwiGLU, Matmul, and RMSNorm over the paper shapes and dtypes.

```bash
python artifact/scripts/run_core.py --figure fig4
```

The runner reads `artifact/configs/fig4_standalone_ops.json` and runs scripts under `tpu_benchmark/`.

### Figure 5: Fused Kernels

Figure 5 covers optimized FlashAttention and MLP W8A16 dequantized forward.

```bash
python artifact/scripts/run_core.py --figure fig5
```

The runner reads `artifact/configs/fig5_fused_ops.json`. FlashAttention fp32 is outside the valid paper comparison and is not part of this matrix.

### Figure 6: Ablation

Figure 6 studies pipeline lowering and bank-aware LMEM placement on representative FlashAttention and Matmul cases.

```bash
python artifact/scripts/run_ablation.py
```

The ablation runner uses existing benchmark scripts and creates temporary stage-0 variants in the same benchmark directory, then removes the temporary files after execution.

The variants are:

| Variant | Meaning |
|---|---|
| `baseline` | Paper path. |
| `pipeline_off` | Rewrites `num_stages=1` or `NUM_STAGES = 1` to stage 0 in a temporary file. |
| `address_off` | Sets `TL_TPU_DISABLE_LIVE_RANGE=1` and `TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1`. |
| `joint_off` | Combines both toggles. |

## Output Format

Each runner writes a directory like:

```text
artifact/results/smoke_<timestamp>/
  environment.json
  raw_results.jsonl
  summary.json
  summary.md
  figure_inputs.csv
  artifact_manifest.json
  logs/
```

`summary.md` gives a readable table. `figure_inputs.csv` is the main input for recreating paper-style plots. `artifact_manifest.json` records files that belong to the run.

Parse a run directory after execution:

```bash
python artifact/scripts/parse_results.py --input artifact/results/<run>/raw_results.jsonl --out-dir artifact/results/<run>/parsed
```

Verify a manifest:

```bash
python artifact/scripts/verify_manifest.py artifact/results/<run>/artifact_manifest.json
```

Copy the submitted PDF figures into a run output directory:

```bash
python artifact/scripts/regenerate_figures.py --out-dir artifact/results/figures_copy
```

This helper copies the submitted figures. It does not recompute performance values.

## Troubleshooting

If `import tilelang` fails, check:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
pip install -e . -v
```

If TVM lowering fails after a fresh checkout, reapply the patch:

```bash
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
cd ../..
```

If PPL-related tests fail with `PPL_PROJECT_ROOT environment variable is not set`, export the PPL SDK path:

```bash
export PPL_PROJECT_ROOT=/path/to/ppl
```

If generated TPU host/device code fails to link, check:

```bash
export SRC_DIR="$PWD/src/tl_templates/tpu"
export TPU_KERNEL_PATH="$SRC_DIR"
export PPL_KERNEL_PATH="$SRC_DIR/libkernel.so"
export LD_LIBRARY_PATH="/opt/tpuv7/tpuv7-current/lib:${PPL_PROJECT_ROOT}/runtime/bm1690/tpuv7-runtime-emulator/lib:${LD_LIBRARY_PATH}"
```

If cmodel is too slow, use the smoke suite only. Cmodel exists to validate correctness without a physical TPU card.

## Files Excluded From Result Archives

Do not include these in an artifact archive:

- `.venv/`
- `build/`
- `artifact/results/`
- `tpu_benchmark/results/`
- generated `src/tl_templates/tpu/kernel.c`, `kernel.cpp`, `kernel.h`, `main.cpp`, `libkernel.so`, `main.so`
- private machine paths, credentials, or board access instructions

## Limitations

FlashAttention fp32 is excluded from performance comparison. Matmul fp32 uses the BM1690 GEMM semantics implemented by this repository. Performance values can shift with device load; reviewer checks should compare correctness, speedup direction, and approximate speedup scale.
