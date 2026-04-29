"""
Run autotune across all tasks in BENCHMARK_RESULTS.md and print a results table.
Run sequentially (one kernel at a time) to avoid TPU card contention.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autotune import autotune_matmul, autotune_rmsnorm, autotune_flashattn

RESULTS = []   # (op, shape, dtype, best_cfg, best_ms)


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─── Matmul ───────────────────────────────────────────────────────────────────
section("MATMUL")
for (M, N, K) in [(64,64,64), (128,128,128), (256,256,256), (512,512,512), (1024,1024,1024)]:
    for dtype in ["float32", "bfloat16", "float16"]:
        atol = 1e-1
        print(f"\n[Matmul {dtype.upper().replace('FLOAT','FP').replace('BFLOAT','BF')} {M}x{N}x{K}]")
        cfg, ms, _ = autotune_matmul(M, N, K, dtype=dtype, atol=atol, rtol=atol)
        key = f"Matmul {dtype} {M}x{N}x{K}"
        RESULTS.append((key, cfg, ms))

# ─── RMSNorm ──────────────────────────────────────────────────────────────────
section("RMSNORM")
for (M, N) in [(64,64), (128,128), (256,256), (512,512), (1024,1024)]:
    for dtype in ["float32", "bfloat16", "float16"]:
        atol = 5e-2 if dtype != "float32" else 1e-2
        print(f"\n[RMSNorm {dtype.upper().replace('FLOAT','FP').replace('BFLOAT','BF')} {M}x{N}]")
        cfg, ms, _ = autotune_rmsnorm(M, N, dtype=dtype, atol=atol, rtol=atol)
        key = f"RMSNorm {dtype} {M}x{N}"
        RESULTS.append((key, cfg, ms))

# ─── FlashAttention ───────────────────────────────────────────────────────────
section("FLASHATTENTION")
fa_configs = [
    (1, 2,  4,   8),
    (1, 8,  64,  32),
    (1, 8,  128, 64),
    (1, 8,  256, 128),
    (1, 8,  512, 128),
]
for (B, H, S, D) in fa_configs:
    for dtype in ["bfloat16", "float16"]:
        atol = 5e-2
        print(f"\n[FlashAttn {dtype.upper().replace('FLOAT','FP').replace('BFLOAT','BF')} B{B}H{H}S{S}D{D}]")
        cfg, ms, _ = autotune_flashattn(B, H, S, D, dtype=dtype, atol=atol, rtol=atol)
        key = f"FlashAttn {dtype} B{B}H{H}S{S}D{D}"
        RESULTS.append((key, cfg, ms))

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  AUTOTUNE SUMMARY")
print("=" * 70)
print(f"{'Task':<45} {'Best Config':<28} {'ms':>8}")
print("-" * 82)
for key, cfg, ms in RESULTS:
    cfg_str = "  ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"{key:<45} {cfg_str:<28} {ms:>8.3f}")
