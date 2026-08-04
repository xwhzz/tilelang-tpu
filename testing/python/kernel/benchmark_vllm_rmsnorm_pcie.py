import ctypes
import os
import statistics
import time

import numpy as np
from tpu_demo.ppl.rms_norm.vllm_sophgo_rmsnorm_forward.vllm_sophgo_rmsnorm_forward import (
    build_kernel,
)


def check_status(name, status):
    if status != 0:
        raise RuntimeError(f"{name} failed with status {status}")


def main():
    m = int(os.environ.get("RMS_M", "32"))
    n = int(os.environ.get("RMS_N", "4096"))
    warmup = int(os.environ.get("RMS_WARMUP", "10"))
    iterations = int(os.environ.get("RMS_ITERS", "100"))
    epsilon = float(os.environ.get("RMS_EPS", "1e-6"))
    kernel = build_kernel(m, n, mode="pcie")

    runtime = ctypes.CDLL("libtpuv7_rt.so")
    runtime.tpuRtInit.restype = ctypes.c_int
    runtime.tpuRtSetDevice.argtypes = [ctypes.c_int]
    runtime.tpuRtSetDevice.restype = ctypes.c_int
    runtime.tpuRtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.tpuRtMalloc.restype = ctypes.c_int
    runtime.tpuRtFree.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    runtime.tpuRtFree.restype = ctypes.c_int
    runtime.tpuRtMemcpyS2D.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.tpuRtMemcpyS2D.restype = ctypes.c_int
    runtime.tpuRtMemcpyD2S.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.tpuRtMemcpyD2S.restype = ctypes.c_int

    check_status("tpuRtInit", runtime.tpuRtInit())
    check_status("tpuRtSetDevice", runtime.tpuRtSetDevice(0))

    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(m, n)).astype(np.float16)
    weight = rng.normal(size=(n,)).astype(np.float16)
    output = np.empty_like(hidden)

    allocations = []
    try:
        for size in (hidden.nbytes, weight.nbytes, output.nbytes):
            ptr = ctypes.c_void_p()
            check_status(
                "tpuRtMalloc",
                runtime.tpuRtMalloc(ctypes.byref(ptr), size, 0),
            )
            allocations.append(ptr)

        hidden_dev, weight_dev, output_dev = allocations
        check_status(
            "tpuRtMemcpyS2D(hidden)",
            runtime.tpuRtMemcpyS2D(
                hidden_dev,
                ctypes.c_void_p(hidden.ctypes.data),
                hidden.nbytes,
            ),
        )
        check_status(
            "tpuRtMemcpyS2D(weight)",
            runtime.tpuRtMemcpyS2D(
                weight_dev,
                ctypes.c_void_p(weight.ctypes.data),
                weight.nbytes,
            ),
        )

        axis = ctypes.c_int32(1)
        eps = ctypes.c_float(epsilon)
        argv = (ctypes.c_void_p * 6)(
            hidden_dev.value,
            weight_dev.value,
            None,
            output_dev.value,
            ctypes.addressof(axis),
            ctypes.addressof(eps),
        )

        for _ in range(warmup):
            check_status(
                "tilelang_tpu_run_device",
                kernel.adapter.lib.tilelang_tpu_run_device(argv),
            )

        samples_ms = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            check_status(
                "tilelang_tpu_run_device",
                kernel.adapter.lib.tilelang_tpu_run_device(argv),
            )
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)

        check_status(
            "tpuRtMemcpyD2S(output)",
            runtime.tpuRtMemcpyD2S(
                ctypes.c_void_p(output.ctypes.data),
                output_dev,
                output.nbytes,
            ),
        )
        reference_fp32 = hidden.astype(np.float32)
        reference_fp32 *= 1.0 / np.sqrt(
            np.mean(reference_fp32 * reference_fp32, axis=-1, keepdims=True)
            + epsilon
        )
        reference = (reference_fp32 * weight.astype(np.float32)).astype(np.float16)
        max_diff = np.max(
            np.abs(output.astype(np.float32) - reference.astype(np.float32))
        )

        print(f"shape={m}x{n}, warmup={warmup}, iterations={iterations}")
        print(f"max_diff={max_diff:.8f}")
        print(
            "kernel_ms: "
            f"avg={statistics.fmean(samples_ms):.6f}, "
            f"min={min(samples_ms):.6f}, "
            f"p50={statistics.median(samples_ms):.6f}, "
            f"max={max(samples_ms):.6f}"
        )
    finally:
        for ptr in allocations:
            runtime.tpuRtFree(ctypes.byref(ptr), 0)


if __name__ == "__main__":
    main()
