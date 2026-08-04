# vLLM Sophgo RMSNorm

The callable follows the `torch.ops.my_ops.rmsnorm_forward` argument order:

```python
kernel(
    hidden_states,
    weight,
    None,
    output,
    hidden_states.dim() - 1,
    variance_epsilon,
)
```

`bias` is reserved for ABI compatibility and is not applied by this vLLM
variant. The implementation normalizes the last dimension.

For BM1690 hardware, compile in PCIe mode. Do not deploy artifacts generated
with `mode="cmodel"`:

```bash
RMS_MODE=pcie RMS_DEVICE=tpu:0 \
python tpu_demo/ppl/rms_norm/vllm_sophgo_rmsnorm_forward/vllm_sophgo_rmsnorm_forward.py
```

The PCIe adapter passes TPU tensor addresses directly to the runtime. CPU
tensors use the host-copy path and are not suitable for latency measurement.
The checked-in C/C++ files show the six-argument ABI for the `32 x 4096`
sample; `libkernel.so` and `main.so` must be regenerated on a machine with the
BM1690 PPL toolchain.

Run the direct-device benchmark with resident device buffers and one
completion synchronization per measured invocation:

```bash
RMS_M=32 RMS_N=4096 RMS_WARMUP=10 RMS_ITERS=100 \
python testing/python/kernel/benchmark_vllm_rmsnorm_pcie.py
```

Run the torch_tpu correctness and latency checks with:

```bash
RMS_MODE=pcie RMS_DEVICE=tpu:0 \
pytest -q testing/python/kernel/test_tilelang_kernel_vllm_rmsnorm.py
```
