# `paged_latent_attention_fp8` 性能测试交接

**更新时间：2026-07-25**

## 1. 远程仓库

- SSH 主机别名：`tpu`
- 用户：`tilelanguser11`
- 仓库：`/mnt2/users/tilelanguser11/tilelang-tpu`
- 分支：`feature/paged-latent-attention-fp8-sophgo-clean`
- 远程：`wyleeyuan`（`wyleeyuan/tilelang-tpu`）
- 不要修改 `main`，不要操作其他用户。

## 2. 两个可独立测试的版本

| 版本 | 仓库文件 | 来源提交 | 已验证平均耗时 |
| --- | --- | --- | ---: |
| 原地累加版 | `tpu_demo/paged_latent_attention_fp8_accumulation.py` | `bd89d41` | `0.9034 ms` |
| 联合 WUQ 版 | `tpu_demo/paged_latent_attention_fp8_joint_wuq.py` | `955ab1b` | `0.7652 ms` |

正式入口 `tpu_demo/paged_latent_attention_fp8.py` 保持为联合 WUQ 最新版。

### 原地累加版

- 利用 PPL GEMM 对输出矩阵 C 的原地累加语义；
- 删除 WUQ/WUV 中 28 组临时 clear/add；
- shared buffer 数量从 71 降到 68；
- 128 rows、`block_n=64`、8 核，5 次预热后统计 10 组：
  平均 `0.9034 ms`，中位数 `0.9015 ms`。

### 联合 WUQ 版

- 将 128 维 `q_nope` 和 64 维 `q_rope` 合并为一次 192 维投影；
- 每个输入块少一次 FP8 转 BF16、反量化缩放和 GEMM dispatch；
- shared buffer 数量从 68 降到 63；
- 相同正式测试配置：平均 `0.7652 ms`，中位数 `0.7535 ms`。

## 3. 已验证正确性

联合 WUQ 版：

- CPU reference 最大误差：`0.000023`
- 输出 finite：通过
- CPU reference allclose：通过
- KV cache：完全一致
- PE cache：完全一致
- 重复调用输出：逐位一致

## 4. 测试环境与注意事项

- 使用 TPU device 14 作为正式性能对比设备。
- device 15 可运行，但同一份原地累加版的 5 组平均为 `0.9398 ms`，
  比 device 14 的 `0.9034 ms` 慢约 `4.0%`。
- device 0 当前为 Fault，不要使用。
- generated host wrapper 默认写入 device 0；现有 benchmark 会在编译时将
  `tpuRtSetDevice(0)` 替换为 `PLA_DEVICE_ID`。
- `HOST_TIMING_MS` 约 32--33 秒是框架调用开销，不是 kernel 时间。
  性能结论应读取日志中的 `Average execution time`。
- 外层 benchmark 输出顺序为：预热组、正式统计组、最后一次重复调用。

## 5. 现有测试入口

- benchmark：`/mnt2/users/tilelanguser11/benchmark_fused_128.py`
- runner：`/mnt2/users/tilelanguser11/run_paged_latent_attention_experiment.sh`
- 公共输入/reference：
  `/mnt2/users/tilelanguser11/pla_benchmark_base.py`

benchmark 已将仓库的 `tpu_demo` 目录加入 Python module 搜索路径，因此可以直接通过
`PLA_MODULE` 选择两个版本。

### 原地累加版短测

```bash
PLA_MODULE=paged_latent_attention_fp8_accumulation \
PLA_DEVICE_ID=14 \
PLA_WARMUP=1 \
PLA_RUNS=5 \
/mnt2/users/tilelanguser11/run_paged_latent_attention_experiment.sh
```

### 联合 WUQ 版短测

```bash
PLA_MODULE=paged_latent_attention_fp8_joint_wuq \
PLA_DEVICE_ID=14 \
PLA_WARMUP=1 \
PLA_RUNS=5 \
/mnt2/users/tilelanguser11/run_paged_latent_attention_experiment.sh
```

正式性能统计将 `PLA_WARMUP=5`、`PLA_RUNS=10`。启用 CPU reference 时再设置：

```bash
PLA_SKIP_REFERENCE=0
```

## 6. 当前性能关系

```text
本轮初始稳定版       1.0255 ms
原地累加版           0.9034 ms   提升 11.9%
联合 WUQ 版          0.7652 ms   再提升 15.3%
累计提升                         25.4%
```
