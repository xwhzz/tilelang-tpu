# FP8 Paged Latent Attention：Decode / Prefill 与 PPL/C 性能差距

日期：2026-08-01

## 结论

| 阶段 | TileLang | PPL/C | TileLang / PPL | 绝对差距 |
| --- | ---: | ---: | ---: | ---: |
| Decode，32 heads，context 128 | 典型 `0.507 ms` | 平均 `0.257661 ms` | `1.968x` | `0.249339 ms` |
| Prefill，16 heads，sequence 192 | 三段合计 `20.470 ms` | 平均 `0.260529 ms` | `78.571x` | `20.209471 ms` |

Decode 已经接近 PPL/C 的 2 倍；Prefill 仍是正确性优先原型，与原版 PPL/C
相差约 78.6 倍。Prefill 的主要差距来自 attention：`19.515 ms`，占当前三段
kernel 合计时间的 `95.33%`。

## 测试环境与口径

- 服务器 HEAD：`f832e36130b4d99ee027a627512ccf5c3f84ac2b`
- TPU device：7，测试后状态 Active、0% utilization、0 MB、TPU clock 1000 MHz
- runtime：`/opt/tpuv7/tpuv7-runtime_1.9.1/lib/libtpuv7_rt.so`
- 计时边界：host steady clock 包围 kernel launch 与 stream synchronize
- Decode 数据来自此前 device 7 正式基线；统一入口改动未改变 Decode kernel
- Prefill PPL/C 与 TileLang 均为 batch 1、16 heads、query length 192、cache
  length 0、page size 16、q rank 1536、kv rank 512、nope 128、RoPE 64、
  value 128、quant block 128、paged mode 2、8 cores

TileLang Prefill 是三个顺序 kernel，下面报告各 kernel 自身的 launch + synchronize；
PPL/C 是一个 fused kernel。TileLang adapter 中 Query RoPE、packing、padding 和 tensor
分配的 host 时间没有计入，因此 `20.470 ms` 已经是对 TileLang 更有利的比较口径。

## Decode

- corrected PPL/C：平均 `0.257661 ms`，中位 `0.252727 ms`
- 当前 TileLang：典型 `0.507 ms`（稳定段约 `0.506–0.508 ms`）
- TileLang 是 PPL/C 的 `1.9677x`
- PPL/C 延迟比 TileLang 低 `49.18%`

这里继续使用 corrected PPL/C：原版 Decode 的 batch-1 输出写回条件有问题，只修复
该条件，attention、FP8 反量化、权重加载和多核计算流程不变。

## Prefill

### 当前 TileLang，16 heads

同一次 device 7 复测中最后一组稳定内部统计：

| kernel | 平均 | 最小 | 最大 |
| --- | ---: | ---: | ---: |
| cache / RoPE / scatter | `0.126 ms` | `0.118 ms` | `0.141 ms` |
| FP8 WUQ/WUKV projection | `0.829 ms` | `0.806 ms` | `0.851 ms` |
| 两遍 attention | `19.515 ms` | `19.318 ms` | `19.792 ms` |
| 平均值合计 | `20.470 ms` | — | — |

正确性同时复核通过：最终输出 max abs `0.081806`、mean abs `0.002313`，
WUQ/WUKV projection、KV/PE cache、未触碰 cache rows 和重复调用均通过。

### 原版 PPL/C Prefill

原版 `mla_prefill.pl` 未修改，使用此前 Decode 基线相同的 PPL 1.6.19 compiler
生成 device C，再构建为 PCIe firmware。正式统计为 10 次预热、100 次运行：

- 平均：`0.260529 ms`
- 中位：`0.254452 ms`
- P10：`0.245635 ms`
- P90：`0.289077 ms`
- 最小：`0.240451 ms`
- 最大：`0.336668 ms`

性能 benchmark 使用零值输入以避免为 PPL 单独转换测试数据布局；该 kernel 不按数值
分支，执行 shape、mask 路径和指令流不变。计时前以 BF16 NaN sentinel 填充输出，
一次调用后 `393216 / 393216` 个输出元素全部被写回，排除了空跑或 batch-1 不写输出。

PPL/C 的等效 kernel-only 吞吐约 `736,962 token/s`；当前 TileLang 三段合计约
`9,380 token/s`。这些只是固定 batch-1、192-token shape 的设备 kernel 吞吐，
不是完整模型吞吐。

## 产物校验

- 原版 Prefill source SHA256：
  `e0f651b3874ac371b1539feb293ec79295e1aed1ad9041ad48bd4a311622699d`
- PPL firmware SHA256：
  `424c4fe1ff9727a5f6079780eaa8b71950faf48d918e608a4788cb9c5655791e`
- PPL hardware benchmark SHA256：
  `441cf23b2d02aa64dafba0d3050cc9d34f92c7f4e6b4ac5670a5c2b792b96b66`
- 当前统一 TileLang source SHA256：
  `d0a8b20a6a132540b440f9d6dfe24bc08232a86060fb2a4fff4ad1f8c850fe71`

Prefill PPL/C benchmark 源码保存在：
`paged_latent_attention_fp8/ppl_vs_tilelang_benchmark/benchmark_prefill_ppl.cpp`。

## 优化含义

Decode 的下一步仍是减少同 core 多 head 的重复权重准备和加强流水化，目标是缩小约
`0.25 ms` 的绝对差距。

Prefill 不应优先微调 cache kernel。attention 当前把每个 query token 当作 `M=1`
处理，Q/K padding 到 512、KV padding 到 256，并扫描两遍 KV；应先实现稳定的
`M>1` query tile、去掉 512/256 padding，并改成单遍 online softmax。只优化
projection 或 cache 最多影响约 `0.955 ms`，无法解决 78.6 倍的主体差距。
