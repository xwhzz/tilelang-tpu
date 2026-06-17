# TileLang-TPU Artifact Guide

This directory is the canonical artifact guide for the paper "Practical: Retargeting AI Kernel DSLs Beyond GPUs: An Experience Report on Refactoring TileLang to Sophgo TPUs". TileLang-TPU is the system under review. The artifact contains install commands, smoke tests, BM1690 PCIe reproduction scripts, cmodel correctness checks, expected paper claims, copied paper figures, result parsers, and manifest checks.

The target badge is **Artifacts Evaluated, Functional**. A **Reusable** review is also possible when the reviewer has access to the SOPHGO BM1690 software stack.

Run all commands below from the repository root.

## Reviewer Path

| Goal | Command | Requires | Expected output |
|---|---|---|---|
| Check local setup | `python artifact/scripts/check_env.py --mode cmodel` | CPU host plus Python deps | Environment report with warnings only for missing hardware paths. |
| Correctness smoke without a TPU card | `python artifact/scripts/run_smoke.py --mode cmodel` | SOPHGO cmodel runtime, no physical card | `artifact/results/smoke_*/summary.md` and manifest. |
| PCIe smoke | `python artifact/scripts/run_smoke.py --mode pcie` | BM1690 PCIe card and PPL SDK | Correctness, timing, logs, manifest. |
| Reproduce Figure 4 | `python artifact/scripts/run_core.py --figure fig4` | BM1690 PCIe card | Standalone operator table in `figure_inputs.csv`. |
| Reproduce Figure 5 | `python artifact/scripts/run_core.py --figure fig5` | BM1690 PCIe card | Fused workload table in `figure_inputs.csv`. |
| Reproduce Figure 6 | `python artifact/scripts/run_ablation.py` | BM1690 PCIe card | Ablation summary and logs. |

Use `--dry-run` on any runner to inspect commands without executing benchmarks.

## Claims and Figures

The paper figures are copied under `artifact/figures/paper/`. The README links them instead of embedding them, because the source files are PDFs and many artifact review renderers do not preview PDFs inline.

| Paper item | Figure file | Expected claim | Reproduction command |
|---|---|---|---|
| Standalone operators | [exp_ops_speedup_one_row.pdf](figures/paper/exp_ops_speedup_one_row.pdf) | Add, RoPE, SwiGLU, Matmul, and RMSNorm average `1.05x` to `1.23x` over PPL. | `python artifact/scripts/run_core.py --figure fig4` |
| Fused workloads | [exp_attention_mlp_speedup_halfcol.pdf](figures/paper/exp_attention_mlp_speedup_halfcol.pdf) | FlashAttention averages `12.50x`; MLP W8A16 averages about `1.01x`. | `python artifact/scripts/run_core.py --figure fig5` |
| Pipeline and address ablation | [ab_speedup_vs_ppl_split.pdf](figures/paper/ablation/ab_speedup_vs_ppl_split.pdf) | Pipeline lowering and bank-aware LMEM placement each contribute positive speedup on representative cases. | `python artifact/scripts/run_ablation.py` |

The machine-readable claim file is [expected/paper_claims.json](expected/paper_claims.json). Correctness tolerances are in [expected/tolerances.yaml](expected/tolerances.yaml).

## Paper Appendix

The required paper appendix is rendered in the repository landing page and stored under `../appendix/`:

- [page-1.png](../appendix/page-1.png)
- [page-2.png](../appendix/page-2.png)
- [page-3.png](../appendix/page-3.png)
- [page-4.png](../appendix/page-4.png)

## Requirements

### PCIe Performance

Performance reproduction requires:

- One SOPHGO BM1690 PCIe TPU card.
- Ubuntu 24.04 LTS.
- SOPHGO TPU driver v1.9.1 and matching firmware.
- SOPHGO PPL SDK v1.4.195.
- Python 3.10 or newer.
- GCC 13.3 or a compatible compiler.
- Xuantie-900 RISC-V toolchain from the SOPHGO/PPL setup.
- PyTorch, NumPy, and Matplotlib.

This artifact does not ship SOPHGO driver, firmware, PPL SDK, or hardware access credentials.

### Cmodel Correctness

Reviewers without a physical BM1690 card can use `mode="cmodel"` on a CPU host with the SOPHGO emulator runtime. Cmodel validates compilation, host/device glue, emulator execution, and output correctness. It does not produce paper performance numbers.

TileLang-TPU defaults to PCIe mode when `mode` is omitted:

```python
tilelang.compile(kernel, out_idx=-1, target="tpu")  # default: PCIe
tilelang.compile(kernel, out_idx=-1, target="tpu", mode="cmodel")
```

## Quick Setup

Create the Python environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-build.txt
pip install -r requirements.txt
pip install -r requirements-artifact.txt
```

Apply the TVM patch and install TileLang-TPU:

```bash
git submodule update --init --recursive
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
cd ../..
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

Then check the environment:

```bash
python artifact/scripts/check_env.py --mode cmodel
python artifact/scripts/check_env.py --mode pcie
```

More setup and troubleshooting details are in [ARTIFACT_APPENDIX.md](ARTIFACT_APPENDIX.md).

## Running the Artifact

### Smoke Tests

```bash
python artifact/scripts/run_smoke.py --mode cmodel
python artifact/scripts/run_smoke.py --mode pcie
```

### Paper Figures

```bash
python artifact/scripts/run_core.py --figure fig4
python artifact/scripts/run_core.py --figure fig5
python artifact/scripts/run_ablation.py
```

To run Figure 4 and Figure 5 together:

```bash
python artifact/scripts/run_full.py
```

### Parse and Verify Results

Each run writes `environment.json`, `raw_results.jsonl`, `summary.json`, `summary.md`, `figure_inputs.csv`, logs, and `artifact_manifest.json`.

```bash
python artifact/scripts/parse_results.py --input artifact/results/<run>/raw_results.jsonl --out-dir artifact/results/<run>/parsed
python artifact/scripts/verify_manifest.py artifact/results/<run>/artifact_manifest.json
```

To copy the submitted PDF figures into a result directory:

```bash
python artifact/scripts/regenerate_figures.py --out-dir artifact/results/figures_copy
```

This helper copies the submitted figures. It does not recompute performance values.

## Acceptance Criteria

- Cmodel runs should pass correctness checks; ignore cmodel timing for paper claims.
- PCIe runs should pass correctness checks and show the same speedup direction and approximate scale as the paper.
- Speedup is `PPL latency / TileLang-TPU latency`; values above `1.0x` mean TileLang-TPU is faster.
- The default timing protocol is 5 warmup runs and 10 measured runs.
- Performance can vary with device load, driver state, and clock behavior. Do not require exact microsecond equality.

## Known Boundaries

- FlashAttention fp32 is excluded from valid TileLang-TPU versus PPL performance comparison. Use fp16 and bf16.
- Matmul fp32 follows the current BM1690 GEMM path: fp32 inputs/outputs, fp16 GEMM operands, and fp32 accumulation semantics.
- If C++ sources under `src/` change, rebuild native components before rerunning: `make -C build -j10`.
- Generated result directories are ignored by git under `artifact/results/`.

## Directory Map

| Path | Role |
|---|---|
| `artifact/README.md` | Main reviewer guide. |
| `artifact/ARTIFACT_APPENDIX.md` | Submission metadata, detailed setup, output format, and troubleshooting. |
| `artifact/CHECKLIST.md` | Artifact review checklist. |
| `artifact/configs/` | Benchmark matrices for smoke, paper figures, and cmodel checks. |
| `artifact/scripts/` | Environment checks, runners, parsers, figure-copy helper, manifest verifier. |
| `artifact/expected/` | Expected paper claims, tolerances, and smoke expectations. |
| `artifact/figures/paper/` | Submitted paper figures copied into the artifact. |
| `appendix/` | Required paper appendix pages rendered as PNG. |
| `tpu_benchmark/` | Benchmark implementations for TileLang-TPU and PPL baselines. |
| `tpu_demo/` | Small TPU and cmodel examples used for diagnostics. |
