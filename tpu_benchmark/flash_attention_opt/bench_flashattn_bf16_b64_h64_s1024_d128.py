"""
Benchmark: optimized FlashAttention bf16_b64_h64_s1024_d128 vs PPL.

FA large shape S1024. Timing uses the TPU template default:
5 warmup runs, then 10 measured runs.
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "fa_large_shape"))

from bench_fa_large_shape import run_fixed


if __name__ == "__main__":
    run_fixed("s1024", "bf16")
