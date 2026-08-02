# FP8 Paged Latent Attention Prefill 多核优化记录（2026-08-02）

## 目标与结论

固定测试范围为 batch 1、sequence 192、mode 2、8 TPU cores，主目标是让
16-head Prefill 的 device kernel 合计延迟不超过原版 PPL/C `0.260529 ms` 的
10 倍，即 `2.60529 ms`。

最终 16-head 四个 device kernel 的三轮 10-run 均值约为：

| 阶段 | 延迟（ms） |
| --- | ---: |
| RoPE + paged cache update | 0.117 |
| FP8 Q/KV projection | 0.489 |
| materialized QK | 0.227 |
| normalized-weight PV | 0.193 |
| 合计 | **1.026** |

因此：

- 相对优化前 TileLang 8-core `20.185 ms`：约 **19.7x speedup**；
- 相对原版 PPL/C `0.260529 ms`：约 **3.94x**；
- 相对目标上限 `2.60529 ms`：有约 **2.54x** 余量。

32-head 回归的四段均值为 `0.121 + 0.678 + 0.364 + 0.293 = 1.456 ms`。

## 实现位置与模型接口

- 正式算子：`tpu_demo/paged_latent_attention_fp8.py`（本地镜像为
  `paged_latent_attention_fp8/paged_latent_attention_fp8.py`）
- 测试：`tpu_demo/paged_latent_attention_fp8/benchmark_prefill_192.py`
- Prefill 与 Decode 仍通过同一个模型入口
  `torch.ops.my_ops.paged_latent_attention_fp8` 调用，并按
  `attention_mode=2/3` 分发到独立后端 kernel。
- Prefill 设备实现拆成 cache、projection、QK、PV 四个 kernel；这比把所有阶段
  强行融合到一个局部内存压力很高的 kernel 更稳定，也符合现有 MLA 的阶段化接口。

## 主要优化

1. 将 attention query tile 从 `M=1` 提升到 `M=32`，把 192 个逐 token 工作项
   合并为 6 个 tile，并沿 head 维分给 8 个 core。
2. FP8 Q 与 KV projection 按每 core 连续 head group 处理，复用 query/KV input
   tile，并用 `K=128` GEMM 累加。
3. QK 单独物化为 BF16 全局张量，避免大融合 kernel 中局部地址复用导致的非有限值。
4. QK 采用与已验证 projection 相同的 `M=32,N=384,K=128,transpose_B=True`
   形状；sequence 192 padding 到 384，padding 权重严格为零。
5. PV 同样 padding 输出维到 384，使用三次
   `M=32,N=384,K=128,transpose_B=True` 累加，最后只写前 128 value dims。
6. Kernel 输出采用二维 head-major `[heads * seq, value_dim]`，适配层再转换为
   模型所需 `[seq, heads, value_dim]`，规避 M>1 下二维 local 到三维 global slice
   的错误写回。

## 数值路径与当前限制

当前 SDK 在 M>1 时暴露了多条不可用路径：FP32/BF16 vector exp、FP32 vector
elementwise、N=1 row-sum/div，以及二维 local 到三维 global 的直接 copy。最终实现
不修改编译器，而是隔离 QK/PV 并只在 device 上使用已验证的 copy/GEMM 形状。

Softmax 权重目前由适配层根据 materialized BF16 QK score 计算四次多项式近似：

`exp(x) ~= (1 + x / 4)^4`

之后施加 causal mask 并按行归一化，再以 BF16 传给 PV kernel。在固定随机模型测试中，
QK score 范围约为 `[-0.06543, 0.05786]`，乘 softmax scale 后绝对值小于
`0.0048`，该近似的最终输出相对 PyTorch 精确 softmax 最大绝对误差为
`0.000244`。这不是通用大-logit softmax 的生产实现；下一阶段应在修复 M>1
vector/reduce 固件路径后，把权重计算下沉到 device，并扩大真实模型分布测试。

测试脚本打印的 `PREFILL_HOST_SYNC_MS` 包含当前生成 host executable 内置的每-kernel
5 次 warmup + 10 次 benchmark，因此不是一次模型调用的真实 host latency；本记录与
原版 PPL 的比较均采用 executable 报告的 device kernel average。

## 正确性与回归

16-head 最终结果：

- Q projection：finite、allclose，max abs `0.000488`；
- KVU projection：finite、allclose，max abs `0.000244`；
- Prefill output：finite、allclose，max abs `0.000244`；
- KV cache：bitwise equal；
- PE cache：allclose，max abs `0.000488`；
- 非 block-table cache rows 未被覆盖；
- 重复 Prefill 输出 bitwise stable。

32-head 最终结果同样通过，且统一 dispatcher 显示 modes `(2, 3)`；Decode mode 3
回归输出 finite。

## 复现命令

```bash
source /mnt2/users/tilelanguser11/envs/tilelang_tpu_dev_v1.4.sh
PLA_DEVICE_ID=7 \
PLA_NUM_HEADS=16 \
PLA_CORE_NUM=8 \
PLA_ATTENTION_BLOCK_M=32 \
PLA_TEST_DECODE=0 \
PLA_WARMUP=0 \
PLA_RUNS=1 \
python /mnt2/users/tilelanguser11/tilelang-tpu/tpu_demo/paged_latent_attention_fp8/benchmark_prefill_192.py
```

32-head + Decode 回归将 `PLA_NUM_HEADS=32`、`PLA_TEST_DECODE=1`。

## 编译器状态

诊断期间曾临时试验新版 exp firmware API；最终已经把
`src/target/codegen_ppl.cc` 恢复为交接前内容，并成功执行
`cmake --build build -j4`。最终优化不依赖编译器修改。
