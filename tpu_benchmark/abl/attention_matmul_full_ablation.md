# BM1690 TileLang Optimization Ablation

Hardware: BM1690. Baseline: TileLang `stages=1`. All cases driven by the bench scripts in `tpu_benchmark/`.

## Ablation Knobs

Controlled via environment variables and one source-level constant:

| Knob | Implementation |
|---|---|
| **Pipeline (stages 1 vs 0)** | Top-level `NUM_STAGES` constant in each `bench_*.py`; switched via `run_matmul_stages.py` or `run_tl_stages_generic.py` |
| **Bank-conflict-aware allocator** | env `TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1` (clears the bank-conflict map; allocator degenerates to pure live-range first-fit) |
| **Live-range buffer reuse** | env `TL_TPU_DISABLE_LIVE_RANGE=1` (forces every buffer's live range to `[0, +∞)`, disabling address sharing) |
| **both OFF** | both env flags set together |

Protocol: baseline taken as the best of one or more runs; the ablated configuration is the worst of several runs. `speedup = baseline_time / ablated_time` (inverted so that values > 1 indicate the disabled optimization carried positive value). When repeated measurements yielded a larger speedup, the older value was replaced.

## Global Summary

### Attention (flash_attention_opt_noPipe, 14 cases)

| Ablation | Overall avg (arithmetic) | bf16 avg | fp16 avg |
|---|---:|---:|---:|
| Address only (live OFF) | 1.034× | 1.044× | 1.024× |
| Address only (both OFF) | **1.027×** | 1.033× | 1.021× |
| Pipeline only (stages=0) | 1.117× | 1.097× | 1.137× |
| **Joint (pipe + addr)** | **1.139×** | 1.142× | 1.137× |

### Matmul (matmul_nopipe, 15 cases)

| Ablation | Overall avg (arithmetic) | bf16 avg | fp16 avg | fp32 avg |
|---|---:|---:|---:|---:|
| Address only (both OFF) | 1.050× | 1.059× | 1.061× | 1.032× |
| Pipeline only (stages=0) | 1.092× | 1.097× | 1.057× | 1.123× |
| **Joint (pipe + addr)** | **1.158×** | 1.191× | 1.124× | 1.160× |

### Cross-Operator Summary

| Dimension | Attention | Matmul |
|---|---:|---:|
| Address alone | +2.7% | +5.0% |
| Pipeline alone | +11.7% | +9.2% |
| Joint | **+13.9%** | **+15.8%** |
| Additivity | joint slightly > pipeline alone; partially independent | pipeline + address partially independent; joint > each alone |

- **Attention** is BDC-bound on the softmax chain; pipeline overlap between V load and softmax accounts for nearly all of the gain. Address allocation contributes almost nothing once `stages=0`.
- **Matmul** carries a heavier DMA share (load A, load B), so both pipeline and address allocation contribute visibly and partially independently.
- **fp32 benefits most on matmul** (+16.0%); **fp16 benefits most on attention** (+13.7%). Both align with the DMA/BDC ratio of each operator/dtype combination.

---

## Reproduction

Change the device id via `tpuRtSetDevice(N)` in `src/tl_templates/tpu/main_template.cpp`.

### Attention

```bash
cd <repo>/tpu_benchmark

# baseline (stages=1)
python run_tl_stages_generic.py \
  flash_attention_opt_noPipe/bench_flashattn_bf16_b1_h8_s512_d128.py 1 baseline

# address off (both flags)
TL_TPU_DISABLE_LIVE_RANGE=1 TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1 \
  python run_tl_stages_generic.py \
    flash_attention_opt_noPipe/bench_flashattn_bf16_b1_h8_s512_d128.py 1 both_off

# joint: pipeline + address off
TL_TPU_DISABLE_LIVE_RANGE=1 TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1 \
  python run_tl_stages_generic.py \
    flash_attention_opt_noPipe/bench_flashattn_bf16_b1_h8_s512_d128.py 0 abl

# fa_large_shape configs (s1024, b8_h8, b8_h32) go through argparse
TL_TPU_DISABLE_LIVE_RANGE=1 TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1 \
  python flash_attention_opt_noPipe/fa_large_shape/bench_fa_large_shape.py \
    --config s1024 --dtype bf16 --backend tl --mode pcie --tl-num-stages 0 \
    --check sample --max-samples 8 --seed 0 --input-pattern v-ramp
```

### Matmul

```bash
cd <repo>/tpu_benchmark

# baseline (stages=1)
python run_matmul_stages.py \
  matmul_nopipe/bench_matmul_bf16_1024x1024x1024.py 1

# pipeline off (stages=0)
python run_matmul_stages.py \
  matmul_nopipe/bench_matmul_bf16_1024x1024x1024.py 0

# address off
TL_TPU_DISABLE_LIVE_RANGE=1 TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1 \
  python run_matmul_stages.py \
    matmul_nopipe/bench_matmul_bf16_1024x1024x1024.py 1

# joint
TL_TPU_DISABLE_LIVE_RANGE=1 TL_TPU_DISABLE_BANK_CONFLICT_AWARE=1 \
  python run_matmul_stages.py \
    matmul_nopipe/bench_matmul_bf16_1024x1024x1024.py 0
```

### Helper script

`run_matmul_stages.py` monkey-patches `bench_module.NUM_STAGES` to flip pipeline stages without editing the source.

```python
import importlib.util, sys, os
BENCH = sys.argv[1]; STAGES = int(sys.argv[2])
spec = importlib.util.spec_from_file_location("bench", os.path.abspath(BENCH))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.NUM_STAGES = STAGES
m.main()
```
