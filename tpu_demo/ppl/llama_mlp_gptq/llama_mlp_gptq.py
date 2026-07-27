# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""GPTQ LLaMA MLP — optimized: merged down quadrant buffers + eliminated redundant zp/scale reloads."""

import tilelang
import tilelang.language as T


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def llama_mlp_gptq(
    M: int,
    K: int,
    group_size: int = 128,
    weight_bits: int = 4,
    block_M: int = 128,
    block_K: int = 128,
    block_B: int = 1,
    num_stages: int = 2,
    threads: int = 128,
    batch_size: int = 1,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    del threads, num_stages
    assert weight_bits == 4

    nepb = 8 // weight_bits
    pm = M // nepb;  pk = K // nepb
    gm = M // group_size;  gk = K // group_size
    pgm = _ceil_div(gm, nepb);  pgk = _ceil_div(gk, nepb)
    bgm = block_M // group_size;  bgk = block_K // group_size
    bpm = block_M // nepb;  bpk = block_K // nepb
    bpgm = _ceil_div(bgm, nepb);  bpgk = _ceil_div(bgk, nepb)
    hk = block_K // 2;  hm = block_M // 2;  hpk = hk // nepb

    assert M % group_size == 0 and K % group_size == 0
    assert M % block_M == 0 and K % block_K == 0
    assert block_M % group_size == 0 and block_K % group_size == 0
    assert batch_size % block_B == 0
    assert block_K % 2 == 0 and block_M % 2 == 0

    ish = (batch_size, M);  osh = (batch_size, M)
    gwsh = (K, pm);  gzsh = (K, pgm);  gssh = (K, gm)
    dwsh = (M, pk);  dzsh = (M, pgk);  dssh = (M, gk)

    ith = (block_B, block_M)
    ahh = (block_B, hk)        # activation half
    ohh = (block_B, hm)        # output half

    whh  = (hk, bpm)           # weight half
    zhh  = (hk, bpgm)          # zp half
    shh  = (hk, bgm)           # scale half
    dqh  = (hk, block_M)       # dequant half
    wkhs = (hk, max(bgm, 2))   # dequant work half shape

    dwh  = (hm, hpk)           # down weight quadrant
    dzh  = (hm, bpgk)          # down zp quadrant
    dsh  = (hm, bgk)           # down scale quadrant
    ddh  = (hm, hk)            # down dequant quadrant
    dkws = (hm, max(bgk, 2))   # down dequant work quadrant shape

    s1 = (1,)

    @T.macro
    def SiLU_Gated_Half(
        gi: T.Tensor(ahh, accum_dtype), ui: T.Tensor(ahh, accum_dtype),
        go: T.Tensor(ahh, accum_dtype),
        ne: T.Tensor(ahh, accum_dtype), de: T.Tensor(ahh, accum_dtype),
        w0: T.Tensor(ahh, accum_dtype), w1: T.Tensor(ahh, accum_dtype),
        co: T.Tensor((64, 32), accum_dtype), tb: T.Tensor((64, 192), accum_dtype),
    ):
        T.ppl_mul_C(ne, gi, T.float32(-1.0))
        T.ppl_exp2(ne, w0, w1, co, tb)
        T.ppl_add_C(de, ne, T.float32(1.0))
        T.ppl_div(ne, gi, de)
        T.ppl_mul(go, ui, ne)

    @T.prim_func
    def mk(
        it: T.Tensor(ish, dtype),
        gw: T.Tensor(gwsh, "uint8"), gz: T.Tensor(gzsh, "uint8"), gs: T.Tensor(gssh, dtype),
        uw: T.Tensor(gwsh, "uint8"), uz: T.Tensor(gzsh, "uint8"), us: T.Tensor(gssh, dtype),
        dw: T.Tensor(dwsh, "uint8"), dz: T.Tensor(dzsh, "uint8"), ds: T.Tensor(dssh, dtype),
        gst: T.Tensor(s1, "int32"), wbt: T.Tensor(s1, "int32"),
        ot: T.Tensor(osh, dtype),
    ):
        with T.Kernel(_ceil_div(M, block_M), _ceil_div(batch_size, block_B), is_cpu=True) as (bx, by):
            ip = T.alloc_shared(ith, dtype)

            # ── gate half 0 ──
            gw0 = T.alloc_shared(whh, "uint8")
            gz0 = T.alloc_shared(zhh, "uint8")
            gs0 = T.alloc_shared(shh, dtype)
            gdq0 = T.alloc_shared(dqh, dtype)
            gwk0 = T.alloc_shared(wkhs, dtype)
            gtmp0 = T.alloc_shared(ahh, accum_dtype)
            ga0 = T.alloc_shared(ahh, accum_dtype)

            # ── gate half 1 ──
            gw1 = T.alloc_shared(whh, "uint8")
            gz1 = T.alloc_shared(zhh, "uint8")
            gs1 = T.alloc_shared(shh, dtype)
            gdq1 = T.alloc_shared(dqh, dtype)
            gwk1 = T.alloc_shared(wkhs, dtype)
            gtmp1 = T.alloc_shared(ahh, accum_dtype)
            ga1 = T.alloc_shared(ahh, accum_dtype)

            # ── up half 0 ──
            uw0 = T.alloc_shared(whh, "uint8")
            uz0 = T.alloc_shared(zhh, "uint8")
            us0 = T.alloc_shared(shh, dtype)
            udq0 = T.alloc_shared(dqh, dtype)
            uwk0 = T.alloc_shared(wkhs, dtype)
            utmp0 = T.alloc_shared(ahh, accum_dtype)
            ua0 = T.alloc_shared(ahh, accum_dtype)

            # ── up half 1 ──
            uw1 = T.alloc_shared(whh, "uint8")
            uz1 = T.alloc_shared(zhh, "uint8")
            us1 = T.alloc_shared(shh, dtype)
            udq1 = T.alloc_shared(dqh, dtype)
            uwk1 = T.alloc_shared(wkhs, dtype)
            utmp1 = T.alloc_shared(ahh, accum_dtype)
            ua1 = T.alloc_shared(ahh, accum_dtype)

            # ── SiLU scratch ──
            ne0 = T.alloc_shared(ahh, accum_dtype)
            de0 = T.alloc_shared(ahh, accum_dtype)
            sw0 = T.alloc_shared(ahh, accum_dtype)
            sw0f = T.alloc_shared(ahh, dtype)
            w00 = T.alloc_shared(ahh, accum_dtype)
            w01 = T.alloc_shared(ahh, accum_dtype)

            ne1 = T.alloc_shared(ahh, accum_dtype)
            de1 = T.alloc_shared(ahh, accum_dtype)
            sw1 = T.alloc_shared(ahh, accum_dtype)
            sw1f = T.alloc_shared(ahh, dtype)
            w10 = T.alloc_shared(ahh, accum_dtype)
            w11 = T.alloc_shared(ahh, accum_dtype)

            cf = T.alloc_shared((64, 32), accum_dtype)
            tb = T.alloc_shared((64, 192), accum_dtype)

            # ── down: merged quadrants ──
            # M-half 0: Q00+Q01 share dw0/dz0/ds0/ddq0/dwk0/dtmp0
            dw0 = T.alloc_shared(dwh, "uint8")
            dz0 = T.alloc_shared(dzh, "uint8")
            ds0 = T.alloc_shared(dsh, dtype)
            ddq0 = T.alloc_shared(ddh, dtype)
            dwk0 = T.alloc_shared(dkws, dtype)
            dtm0 = T.alloc_shared(ohh, accum_dtype)

            # M-half 1: Q10+Q11 share dw1/dz1/ds1/ddq1/dwk1/dtmp1
            dw1 = T.alloc_shared(dwh, "uint8")
            dz1 = T.alloc_shared(dzh, "uint8")
            ds1 = T.alloc_shared(dsh, dtype)
            ddq1 = T.alloc_shared(ddh, dtype)
            dwk1 = T.alloc_shared(dkws, dtype)
            dtm1 = T.alloc_shared(ohh, accum_dtype)

            # ── output halves ──
            oa0 = T.alloc_shared(ohh, accum_dtype)
            oa1 = T.alloc_shared(ohh, accum_dtype)
            oc0 = T.alloc_shared(ohh, dtype)
            oc1 = T.alloc_shared(ohh, dtype)

            T.ppl_fill(oa0, 0.0);  T.ppl_fill(oa1, 0.0)

            for ko in range(_ceil_div(K, block_K)):
                T.ppl_fill(ga0, 0.0);  T.ppl_fill(ga1, 0.0)
                T.ppl_fill(ua0, 0.0);  T.ppl_fill(ua1, 0.0)

                for ki in range(_ceil_div(M, block_M)):
                    T.ppl_copy(it[by * block_B, ki * block_M], ip)

                    # Gate h0: K[0:hk]
                    T.ppl_copy(gw[ko * block_K + 0, ki * bpm], gw0)
                    T.ppl_copy(gz[ko * block_K + 0, (ki * bgm) // 2], gz0)
                    T.ppl_copy(gs[ko * block_K + 0, ki * bgm], gs0)
                    T.ppl_dequant_gptq(gdq0, gw0, gz0, gs0, gwk0,
                                       group_size=group_size, weight_bits=weight_bits,
                                       group_offset=ki * bgm)
                    T.ppl_gemm(ip, gdq0, gtmp0, transpose_B=True, add_result=False)
                    T.ppl_add(ga0, ga0, gtmp0)

                    # Gate h1: K[hk:block_K]
                    T.ppl_copy(gw[ko * block_K + hk, ki * bpm], gw1)
                    T.ppl_copy(gz[ko * block_K + hk, (ki * bgm) // 2], gz1)
                    T.ppl_copy(gs[ko * block_K + hk, ki * bgm], gs1)
                    T.ppl_dequant_gptq(gdq1, gw1, gz1, gs1, gwk1,
                                       group_size=group_size, weight_bits=weight_bits,
                                       group_offset=ki * bgm)
                    T.ppl_gemm(ip, gdq1, gtmp1, transpose_B=True, add_result=False)
                    T.ppl_add(ga1, ga1, gtmp1)

                    # Up h0: K[0:hk]
                    T.ppl_copy(uw[ko * block_K + 0, ki * bpm], uw0)
                    T.ppl_copy(uz[ko * block_K + 0, (ki * bgm) // 2], uz0)
                    T.ppl_copy(us[ko * block_K + 0, ki * bgm], us0)
                    T.ppl_dequant_gptq(udq0, uw0, uz0, us0, uwk0,
                                       group_size=group_size, weight_bits=weight_bits,
                                       group_offset=ki * bgm)
                    T.ppl_gemm(ip, udq0, utmp0, transpose_B=True, add_result=False)
                    T.ppl_add(ua0, ua0, utmp0)

                    # Up h1: K[hk:block_K]
                    T.ppl_copy(uw[ko * block_K + hk, ki * bpm], uw1)
                    T.ppl_copy(uz[ko * block_K + hk, (ki * bgm) // 2], uz1)
                    T.ppl_copy(us[ko * block_K + hk, ki * bgm], us1)
                    T.ppl_dequant_gptq(udq1, uw1, uz1, us1, uwk1,
                                       group_size=group_size, weight_bits=weight_bits,
                                       group_offset=ki * bgm)
                    T.ppl_gemm(ip, udq1, utmp1, transpose_B=True, add_result=False)
                    T.ppl_add(ua1, ua1, utmp1)

                SiLU_Gated_Half(ga0, ua0, sw0, ne0, de0, w00, w01, cf, tb)
                SiLU_Gated_Half(ga1, ua1, sw1, ne1, de1, w10, w11, cf, tb)
                T.ppl_copy(sw0, sw0f)
                T.ppl_copy(sw1, sw1f)

                # ── Down: 4 quadrants, M-half 0+1 each share buffers ──
                # Q00: M[0:hm] × K[0:hk] (load zp/scale once)
                T.ppl_copy(dw[bx * block_M + 0, ko * bpk + 0], dw0)
                T.ppl_copy(dz[bx * block_M + 0, (ko * bgk) // 2], dz0)
                T.ppl_copy(ds[bx * block_M + 0, ko * bgk], ds0)
                T.ppl_dequant_gptq(ddq0, dw0, dz0, ds0, dwk0,
                                   group_size=group_size, weight_bits=weight_bits,
                                   group_offset=ko * bgk)
                T.ppl_gemm(sw0f, ddq0, dtm0, transpose_B=True, add_result=False)
                T.ppl_add(oa0, oa0, dtm0)

                # Q01: M[0:hm] × K[hk:block_K] (reuse dz0/ds0, only reload weight)
                T.ppl_copy(dw[bx * block_M + 0, ko * bpk + hpk], dw0)
                T.ppl_dequant_gptq(ddq0, dw0, dz0, ds0, dwk0,
                                   group_size=group_size, weight_bits=weight_bits,
                                   group_offset=ko * bgk)
                T.ppl_gemm(sw1f, ddq0, dtm0, transpose_B=True, add_result=False)
                T.ppl_add(oa0, oa0, dtm0)

                # Q10: M[hm:block_M] × K[0:hk]
                T.ppl_copy(dw[bx * block_M + hm, ko * bpk + 0], dw1)
                T.ppl_copy(dz[bx * block_M + hm, (ko * bgk) // 2], dz1)
                T.ppl_copy(ds[bx * block_M + hm, ko * bgk], ds1)
                T.ppl_dequant_gptq(ddq1, dw1, dz1, ds1, dwk1,
                                   group_size=group_size, weight_bits=weight_bits,
                                   group_offset=ko * bgk)
                T.ppl_gemm(sw0f, ddq1, dtm1, transpose_B=True, add_result=False)
                T.ppl_add(oa1, oa1, dtm1)

                # Q11: M[hm:block_M] × K[hk:block_K] (reuse dz1/ds1, only reload weight)
                T.ppl_copy(dw[bx * block_M + hm, ko * bpk + hpk], dw1)
                T.ppl_dequant_gptq(ddq1, dw1, dz1, ds1, dwk1,
                                   group_size=group_size, weight_bits=weight_bits,
                                   group_offset=ko * bgk)
                T.ppl_gemm(sw1f, ddq1, dtm1, transpose_B=True, add_result=False)
                T.ppl_add(oa1, oa1, dtm1)

            T.ppl_copy(oa0, oc0)
            T.ppl_copy(oa1, oc1)
            T.ppl_copy(oc0, ot[by * block_B, bx * block_M + 0])
            T.ppl_copy(oc1, ot[by * block_B, bx * block_M + hm])

    return mk


def reference_llama_mlp_gptq(
    input_tensor,
    gate_w, gate_zp, gate_scale,
    up_w, up_zp, up_scale,
    down_w, down_zp, down_scale,
    group_size: int = 128,
):
    import torch
    import torch.nn.functional as F

    def dq(pw, pz, sc):
        rs, pc = pw.shape;  c = pc * 2
        w = torch.empty((rs, c), dtype=torch.float16, device=pw.device)
        for cl in range(c):
            w[:, cl] = ((pw[:, cl // 2] >> (4 * (cl % 2))) & 0xF).to(torch.float16)
        ng = sc.shape[1]
        zp = torch.empty((rs, ng), dtype=torch.float16, device=pz.device)
        for g in range(ng):
            zp[:, g] = ((pz[:, g // 2] >> (4 * (g % 2))) & 0xF).to(torch.float16)
        gsv = c // ng
        ze = torch.empty((rs, c), dtype=torch.float16, device=pw.device)
        se = torch.empty((rs, c), dtype=torch.float16, device=pw.device)
        for g in range(ng):
            s = g * gsv;  e = s + gsv
            ze[:, s:e] = zp[:, g:g+1];  se[:, s:e] = sc[:, g:g+1]
        return (w - ze) * se

    g = dq(gate_w, gate_zp, gate_scale)
    u = dq(up_w, up_zp, up_scale)
    d = dq(down_w, down_zp, down_scale)
    sw = F.silu(F.linear(input_tensor, g).float()) * F.linear(input_tensor, u).float()
    return F.linear(sw.to(torch.float16), d)


if __name__ == "__main__":
    k = llama_mlp_gptq(M=4096, K=11008, group_size=128, weight_bits=4, block_M=128, block_K=128, block_B=1, batch_size=1)
    print(tilelang.lower(k))
