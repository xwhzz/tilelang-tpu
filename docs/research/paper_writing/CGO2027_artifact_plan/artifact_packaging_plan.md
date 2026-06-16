# Practical: Retargeting AI Kernel DSLs Beyond GPUs Artifact 整理规划

本文用于指导 `CGO2027-submission` 分支为论文 `Practical: Retargeting AI Kernel DSLs Beyond GPUs: An Experience Report on Refactoring TileLang to Sophgo TPUs` 整理 artifact。目标不是重新组织整篇论文，而是把当前 TileLang-TPU 仓库整理成 AE reviewer 可以安装、运行、核对论文主结果的工程 artifact。

当前结论：仓库已经包含论文所需的大部分代码、benchmark 脚本、PPL baseline 文件和历史结果；还缺少一个面向 reviewer 的统一 artifact 入口、环境检查、分层运行脚本、结果解析脚本、论文图表到脚本的映射表，以及对 BM1690 专有依赖和已知验证边界的说明。

## 参考依据

| 来源 | 本计划采用的规则 |
|---|---|
| CGO 2027 CFP: <https://conf.researchr.org/track/cgo-2027/cgo-2027-papers> | Practical experience papers 鼓励提交 artifact；AE 负责检查 artifact 对论文工作的支撑。Tool paper 强制通过 AE，practical paper 可能被 conditional acceptance 要求通过 AE。 |
| CGO 2026 AE: <https://2026.cgo.org/track/cgo-2026-artifact-evaluation> | 近年 CGO artifact 采用 ACM badge 口径，重点检查 artifact 是否 functional、reusable、是否支撑论文 claim。 |
| ACM Artifact Review and Badging v1.1: <https://www.acm.org/publications/policies/artifact-review-and-badging-current> | `Functional` 要求 documented、consistent、complete、exercisable，并有 verification and validation 证据；`Reusable` 要求在 functional 基础上文档和结构足够支持复用。 |
| cTuning AE submission guide: <https://ctuning.org/ae/submission-20201122.html> | Artifact abstract 应说明软硬件要求、支撑哪些论文结果、如何验证、预期输出；需要列出 proprietary software/hardware。 |
| cTuning AE reviewing guide: <https://ctuning.org/ae/reviewing-20201122.html> | reviewer 会按 artifact 是否公开、是否 functional、是否完整、是否可运行、是否一致、是否可复用、是否能验证主结果来打分。 |

## Artifact 目标

### 推荐申请 badge

| Badge | 建议 | 原因 |
|---|---|---|
| Artifacts Evaluated, Functional | 必须争取 | 论文 claim 依赖 BM1690 真实运行、PPL baseline、PyTorch correctness reference 和 ablation。当前仓库已有可运行基础。 |
| Artifacts Evaluated, Reusable | 作为 stretch goal | 如果补齐统一 runner、配置化 benchmark、结果解析和文档，TileLang-TPU 可被 reviewer 用于新增 TPU kernel 或新 shape。 |
| Artifacts Available | 录用后争取 | 需要把最终 artifact 放到 Zenodo、Figshare、Dryad 或有长期保存计划的仓库并给 DOI。GitHub 或匿名仓库只能作为 AE 期间访问入口，不能单独支撑该 badge。 |
| Results Reproduced | 不主动承诺 | 该 badge 需要第三方使用 artifact 获得论文主结果。我们可以提供脚本支持，但不能由作者自己声明完成。 |

### 最小通过标准

- reviewer 能从 fresh checkout 完成安装，或在我们提供的 BM1690 机器上完成环境确认。
- reviewer 能运行一个 10 到 30 分钟的 smoke suite，确认编译、JIT、PPL baseline、PyTorch correctness reference 和结果解析都工作。
- reviewer 能运行 core suite，重现论文 Figure 4、Figure 5 和 Figure 6 的关键趋势，允许硬件抖动范围内的数值差异。
- artifact 明确说明哪些结果需要 BM1690 PCIe 模式，哪些可以用 `cmodel` 做功能检查。
- 所有专有依赖，包括 SOPHGO driver、firmware、PPL SDK 和 BM1690 设备访问方式，都写入 artifact abstract 和 README。

## 当前论文 Claim 到 Artifact 的映射

论文目录：`docs/research/paper_writing/Tilelang_TPU_CGO_submitted`

| 论文位置 | Claim 或结果 | 当前仓库证据 | Artifact 状态 | 需要补齐 |
|---|---|---|---|---|
| Abstract, Section 5.2 | standalone operators 平均 `1.05x` 到 `1.23x` over PPL | `tpu_benchmark/performance_summary.md`，`tpu_benchmark/{elementwise_add,rope,swiglu,matmul,rmsnorm_opt}` | 已有结果和多数脚本 | 统一 runner，结果 JSON，图表重生成脚本，论文表格校验 |
| Figure 4 | Add、RoPE、SwiGLU、Matmul、RMSNorm shape/dtype speedup | `tpu_benchmark/*/bench_*.py`，对应 `pl/*.pl` | 分散可运行 | 建 `artifact/run_core.py --figure fig4`，输出 CSV/JSON/PDF |
| Figure 5 | FlashAttention `12.50x`，MLP W8A16 约 `1.01x` | `tpu_benchmark/flash_attention_opt`，`tpu_benchmark/mlp_w8a16_dq_forward` | 已有结果 | 统一运行入口，明确 fp16/bf16 范围，MLP expected output |
| Section 5.3, Figure 6 | structured pipeline lowering 与 bank-aware LMEM placement ablation | `tpu_benchmark/abl/attention_matmul_full_ablation.md`，`plot_speedup_split.py` | 有总结和画图脚本 | 当前树未找到文档中引用的 `run_matmul_stages.py`、`run_tl_stages_generic.py`、`flash_attention_opt_noPipe`、`matmul_nopipe`，需要补回或更新复现说明 |
| Appendix B | operator shapes | `sections/A_appendix.tex` | 已列出 | 将 shape matrix 提取成 `artifact/configs/*.yaml` |
| Appendix C | correctness and performance protocol | `sections/A_appendix.tex`，benchmark 脚本中的 `torch.allclose` | 局部存在 | 汇总 tolerance、warmup/repeat、seed、跳过项 |
| Appendix E | AddressAssign 与 pipeline 实现细节 | `src/transform/address_assign.cc`，`src/transform/pipeline_planning.cc`，`src/transform/inject_pipeline.cc` | 代码存在 | 添加编译 pass smoke 和 address dump 收集脚本 |
| Appendix F | JIT runtime and toolchain flow | `tilelang/jit/adapter/`，`src/tl_templates/tpu/` | 代码存在 | 添加 artifact 中的 generated source 保存和检查命令 |

## 推荐 Artifact 目录结构

建议在仓库根目录新增 `artifact/` 作为 reviewer 入口。当前 `docs/research/paper_writing/CGO2027_artifact_plan/` 只保存规划，不作为运行入口。

```text
artifact/
  README.md
  ARTIFACT_APPENDIX.md
  CHECKLIST.md
  configs/
    environment_bm1690.yaml
    fig4_standalone_ops.yaml
    fig5_fused_ops.yaml
    fig6_ablation.yaml
    smoke.yaml
  scripts/
    check_env.py
    run_smoke.py
    run_core.py
    run_full.py
    run_ablation.py
    collect_env.py
    parse_results.py
    regenerate_figures.py
    verify_manifest.py
  expected/
    smoke_expected.json
    paper_claims.json
    tolerances.yaml
  results/
    .gitignore
  figures/
    .gitignore
```

根目录建议新增或更新：

```text
ARTIFACT.md
requirements-artifact.txt
environment-artifact.yml
```

## Reviewer-facing README 应包含

- Artifact abstract：说明 TileLang-TPU 是 BM1690 上的 TileLang-to-TPU compiler/runtime artifact。
- Badge request：优先 `Artifacts Evaluated, Functional`，可选 `Reusable`。
- Hardware requirement：单张 SOPHGO BM1690 PCIe 卡，Ubuntu 24.04 LTS，driver v1.9.1，对应 firmware，PPL SDK v1.4.195。
- Proprietary dependency statement：SOPHGO driver、firmware、PPL SDK、BM1690 板卡不随 artifact 分发；说明 reviewer 如何获得或使用作者提供的机器。
- Software requirement：Python 3.10、TVM v0.17.dev0 plus `patches/tvm.patch`、GCC 13.3、Xuantie-900-gcc、PyTorch、Python 包依赖。
- Install commands：包含 submodule、TVM patch、`./install_tpu.sh`、`pip install -e . -v`。
- Quick test：一个不超过 30 分钟的命令，跑 Matmul FP16、RMSNorm FP16、RoPE FP16、FlashAttention small shape。
- Full reproduction：列出复现 Figure 4、Figure 5、Figure 6 的命令、预计运行时间、输出路径。
- Expected results：给出 speedup 范围和允许波动，不要求逐微秒完全一致。
- Known limitations：BM1690 专有硬件、FlashAttention fp32 不作为有效 TL/PPL 对比、Matmul fp32 reference 采用当前 BM1690 GEMM 输入语义、板卡负载会导致 latency 抖动。
- Troubleshooting：`tpuRtSetDevice`、`LD_LIBRARY_PATH`、PPL SDK 路径、TVM patch、JIT 生成目录、长时间 D 状态进程处理方式。

## 分层运行设计

| Suite | 目标 | 内容 | 时间预算 | 输出 |
|---|---|---|---:|---|
| `smoke` | 检查 artifact functional | 1 到 2 个 dtype/shape，覆盖 JIT、PPL baseline、correctness、result parser | 10 到 30 分钟 | `results/smoke_<timestamp>/manifest.json` |
| `core` | 支撑论文主结论 | Figure 4 standalone operators，Figure 5 fused ops，Figure 6 ablation 的代表配置 | 2 到 6 小时 | raw JSON、summary CSV、paper table、figures |
| `full` | 重跑论文完整矩阵 | Appendix B 全部 shape/dtype，重复论文 warmup/repeat | 过夜 | full raw logs、summary、regenerated figures |
| `diagnostic` | 帮 reviewer 定位问题 | 编译-only、cmodel smoke、AddressAssign dump、generated `kernel.c` 保存 | 10 到 60 分钟 | logs、generated sources、environment JSON |

## 结果文件规范

每次运行生成一个独立目录：

```text
artifact/results/<suite>_<YYYYMMDD_HHMMSS>/
  environment.json
  matrix.yaml
  raw_results.jsonl
  summary.json
  summary.md
  figure_inputs.csv
  artifact_manifest.json
  logs/
  generated_sources/
```

`raw_results.jsonl` 每行至少包含：

```json
{
  "suite": "core",
  "figure": "fig4",
  "operator": "matmul",
  "dtype": "fp16",
  "shape": "512x512x512",
  "backend": "tilelang",
  "mode": "pcie",
  "correct": true,
  "max_diff": 0.0,
  "avg_us": 7283.0,
  "warmup": 5,
  "repeats": 10,
  "seed": 0,
  "status": "pass"
}
```

`artifact_manifest.json` 需要列出所有 raw logs、summary、generated sources、figures 和环境文件。可以复用 `docs/research/pipeline_optimization/logs/2026-06-04_phase19_paper_eval_artifacts/smoke/artifact_manifest.json` 的形式。

## Checklist

### P0: 冻结范围和提交

- [ ] 记录 artifact 使用的 git commit，例如当前分支头 `16e440cbd6a67f96e90b170882f3ff6dacdc5f29`。
- [ ] 确认 `CGO2027-submission` 与需要包含的 `refactor/op-renaming` 进度一致。
- [ ] 冻结论文版本：`docs/research/paper_writing/Tilelang_TPU_CGO_submitted/main.pdf`。
- [ ] 从论文中提取 Figure 4、Figure 5、Figure 6、Appendix B、Appendix C 的 claim matrix。
- [ ] 标出不进入 artifact claim 的历史结果和临时实验。

### P1: Packaging 基础文件

- [ ] 新增根目录 `ARTIFACT.md`，作为 reviewer 的唯一入口。
- [ ] 新增 `artifact/README.md`，包含 artifact abstract、badge request、硬件、软件、安装、运行、预期输出、限制。
- [ ] 新增 `artifact/ARTIFACT_APPENDIX.md`，可压缩成 camera-ready 的 artifact appendix。
- [ ] 新增 `artifact/CHECKLIST.md`，按 ACM/cTuning checklist 填写。
- [ ] 新增 `requirements-artifact.txt` 或 `environment-artifact.yml`。
- [ ] 确认 `artifact/results/`、`artifact/figures/`、`tpu_benchmark/results/` 都被 git ignore。

### P2: 环境检查和安装脚本

- [ ] 写 `artifact/scripts/check_env.py`，检查 Python、PyTorch、TileLang import、TVM patch、PPL SDK、driver、BM1690 可见性。
- [ ] 写 `artifact/scripts/collect_env.py`，输出 `environment.json`。
- [ ] 明确 `PPL_PROJECT_ROOT`、`LD_LIBRARY_PATH`、SOPHGO SDK 路径和 Xuantie toolchain 路径。
- [ ] 给出 `pcie` 和 `cmodel` 两套运行说明。
- [ ] 对无 BM1690 的 reviewer 提供 compile-only 或 `cmodel` smoke，不把它写成性能复现。

### P3: Benchmark runner

- [ ] 把 Figure 4 standalone ops 配置写入 `artifact/configs/fig4_standalone_ops.yaml`。
- [ ] 把 Figure 5 fused ops 配置写入 `artifact/configs/fig5_fused_ops.yaml`。
- [ ] 把 Figure 6 ablation 配置写入 `artifact/configs/fig6_ablation.yaml`。
- [ ] 写 `artifact/scripts/run_smoke.py`，覆盖 Matmul、RMSNorm、RoPE 或 Add、FlashAttention small shape。
- [ ] 写 `artifact/scripts/run_core.py --figure fig4|fig5|fig6|all`。
- [ ] 写 `artifact/scripts/run_full.py`，重跑论文完整矩阵。
- [ ] 写 `artifact/scripts/run_ablation.py`，统一 toggles：pipeline off、live-range reuse off、bank-conflict-aware off、joint off。
- [ ] 修复或更新 `tpu_benchmark/abl/attention_matmul_full_ablation.md` 中当前树上找不到的 runner 路径。

### P4: Correctness 和统计口径

- [ ] 汇总所有 operator 的 `torch.allclose` tolerance 到 `artifact/expected/tolerances.yaml`。
- [ ] 统一 warmup=5、repeats=10，除非某个 suite 明确标注不同设置。
- [ ] 固定随机 seed，并在输出中记录 seed。
- [ ] Matmul fp32 reference 按 BM1690 当前 GEMM 输入语义处理：fp32 输入输出、fp16 GEMM 操作数、fp32 累加。
- [ ] FlashAttention fp32 明确跳过 TL/PPL 对比，不写 full-fp32 correctness 或 speedup。
- [ ] 对 performance speedup 给出允许波动，例如 small kernel 用宽容区间，大 kernel 用主趋势不变作为判定。

### P5: 结果解析和图表重生成

- [ ] 写 `artifact/scripts/parse_results.py`，从 raw logs 生成 `summary.json`、`summary.md`、`figure_inputs.csv`。
- [ ] 写 `artifact/scripts/regenerate_figures.py`，从 `figure_inputs.csv` 生成 Figure 4、Figure 5、Figure 6 的 PDF。
- [ ] 把当前论文 figure 对应的源数据保存到 `artifact/expected/paper_claims.json`。
- [ ] 检查 regenerated figures 与论文图的 shape、dtype、speedup 口径一致。
- [ ] 保留 raw stdout/stderr，不把日志写到源码目录。

### P6: 文档和 reviewer 操作路径

- [ ] 在 README 顶部放三条命令：环境检查、smoke、core。
- [ ] 每个命令给出预计时间和输出路径。
- [ ] 写 troubleshooting：TVM patch 未应用、PPL 编译失败、找不到 BM1690、device id、JIT 卡住、精度差异。
- [ ] 写 known limitations 和 proprietary dependency statement。
- [ ] 写 support protocol：AE 期间 reviewer 如果没有 BM1690，应联系 AE chairs 或使用作者提供的远程机器。

### P7: Archive 和发布

- [ ] AE 期间使用匿名仓库或匿名 tarball，避免泄露作者身份。
- [ ] 录用后生成公开 release tarball。
- [ ] 把公开 artifact 上传到 Zenodo、Figshare、Dryad 或机构仓库，获取 DOI。
- [ ] 在 camera-ready artifact appendix 中填入 DOI 和最终 commit。
- [ ] 确认不包含 `.venv/`、`build/`、大型临时日志、私有路径、API key、板卡账号信息。

## 当前高风险缺口

| 风险 | 影响 | 推荐动作 |
|---|---|---|
| BM1690 和 SOPHGO SDK 是专有依赖 | reviewer 可能无法本地复现性能 | 提供远程机器或提前联系 AE chairs；README 中明确硬件和授权边界。 |
| 当前没有统一 artifact runner | reviewer 需要手工拼命令 | 新增 `artifact/scripts/run_smoke.py`、`run_core.py`、`run_full.py`。 |
| Ablation 文档引用的部分 runner 当前树上缺失 | Figure 6 复现路径不闭合 | 补回 runner，或把 ablation 复现实验改写到当前实际目录。 |
| `tpu_benchmark/performance_summary.md` 是人工汇总 | 图表不能由 raw data 自动重生 | 统一 raw JSON 到 summary 到 figure 的链路。 |
| FlashAttention fp32 和 Matmul fp32 语义有边界 | reviewer 可能误读 correctness | 在 README、tolerances 和 expected claims 中写清跳过项和 reference 语义。 |
| 板卡性能有抖动 | 微秒级 latency 不能逐值复现 | 用 speedup 区间、平均值、重复次数和主趋势判定。 |

## 建议完成顺序

1. 先补 `ARTIFACT.md`、`artifact/README.md`、`check_env.py` 和 `run_smoke.py`，确保 functional badge 的入口闭合。
2. 再把 Figure 4 和 Figure 5 接入 `run_core.py`，产出 raw JSON、summary 和 figure inputs。
3. 单独处理 Figure 6 ablation，先修复缺失 runner，再重跑代表配置。
4. 最后做 `run_full.py`、图表重生成和 archive DOI。

完成 P1 到 P4 后，artifact 可以进入 AE smoke 预审；完成 P5 到 P7 后，再申请 reusable 和 available。
