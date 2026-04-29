#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autotune import autotune_rmsnorm

results = {}
for M,N in [(64,64),(128,128),(256,256),(512,512),(1024,1024)]:
    for dtype in ['float32','bfloat16','float16']:
        atol = 1e-2 if dtype == 'float32' else 5e-2
        print(f"\n[RMSNorm {dtype} {M}x{N}]", flush=True)
        try:
            cfg, ms, _ = autotune_rmsnorm(M, N, dtype=dtype, atol=atol, rtol=atol, verbose=False)
            results[f"RMSNorm {dtype} {M}x{N}"] = (cfg, ms)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

print("\n=== RMSNORM AUTOTUNE SUMMARY ===", flush=True)
for k, (cfg, ms) in results.items():
    print(f"{k}: {cfg}  {ms:.3f}ms", flush=True)
