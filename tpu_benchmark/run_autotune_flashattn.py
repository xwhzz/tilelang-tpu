#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autotune import autotune_flashattn

fa_configs = [
    (1, 2,   4,   8),
    (1, 8,  64,  32),
    (1, 8, 128,  64),
    (1, 8, 256, 128),
    (1, 8, 512, 128),
]

results = {}
for B,H,S,D in fa_configs:
    for dtype in ['bfloat16','float16']:
        print(f"\n[FlashAttn {dtype} B{B}H{H}S{S}D{D}]", flush=True)
        try:
            cfg, ms, _ = autotune_flashattn(B, H, S, D, dtype=dtype, atol=5e-2, rtol=5e-2, verbose=False)
            results[f"FlashAttn {dtype} B{B}H{H}S{S}D{D}"] = (cfg, ms)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

print("\n=== FLASHATTN AUTOTUNE SUMMARY ===", flush=True)
for k, (cfg, ms) in results.items():
    print(f"{k}: {cfg}  {ms:.3f}ms", flush=True)
