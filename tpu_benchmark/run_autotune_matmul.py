#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autotune import autotune_matmul

results = {}
for M,N,K in [(64,64,64),(128,128,128),(256,256,256),(512,512,512),(1024,1024,1024)]:
    for dtype in ['float32','bfloat16','float16']:
        print(f"\n[Matmul {dtype} {M}x{N}x{K}]", flush=True)
        try:
            cfg, ms, _ = autotune_matmul(M, N, K, dtype=dtype, atol=1e-1, rtol=1e-1, verbose=False)
            results[f"Matmul {dtype} {M}x{N}x{K}"] = (cfg, ms)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

print("\n=== MATMUL AUTOTUNE SUMMARY ===", flush=True)
for k, (cfg, ms) in results.items():
    print(f"{k}: {cfg}  {ms:.3f}ms", flush=True)
