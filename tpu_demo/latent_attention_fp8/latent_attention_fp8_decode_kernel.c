#include "ppl_helper.h"
#ifndef __PPL_HELPER__
#endif  // __PPL_HELPER__

typedef struct {
    dim4 shape;
    dim4 stride;
    global_addr_t addr;
    data_type_t dtype;
    int mode;
    int align_mode;
    int size;
    int offset;
    bool unsigned_flag;
    bool default_stride;
} __ppl_tensor_info;

void latent_attention_fp8_decode_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11, global_addr_t v12, global_addr_t v13, global_addr_t v14, global_addr_t v15, int v16) {
  __ppl_tensor_info v31 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v15, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v30 = {.shape = {1, 192, 1, 512} , .stride = {0}, .addr = v14, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 196608, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {1, 1, 1, 512} , .stride = {0}, .addr = v2, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 1024, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v22 = {.shape = {1, 24, 1, 12} , .stride = {0}, .addr = v6, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {1, 3072, 1, 1536} , .stride = {0}, .addr = v4, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 4718592, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1, 4096, 1, 512} , .stride = {0}, .addr = v5, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 2097152, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v27 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v11, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v17 = {.shape = {1, 1, 1, 1536} , .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 3072, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v29 = {.shape = {1, 1, 1, 2048} , .stride = {0}, .addr = v13, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v24 = {.shape = {1, 192, 1, 512} , .stride = {0}, .addr = v8, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 196608, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v25 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v9, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v26 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v10, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v23 = {.shape = {1, 32, 1, 4} , .stride = {0}, .addr = v7, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v28 = {.shape = {1, 1, 1, 192} , .stride = {0}, .addr = v12, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info q_upper_acc_h0 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0.stride, 0, &q_upper_acc_h0.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0.stride, 0, &q_upper_h0.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1.stride, 0, &q_upper_acc_h1.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1.stride, 0, &q_upper_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0.stride, 0, &q_rope_src_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0.stride, 0, &q_rope_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0.stride, 0, &q_rope_tmp0_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0.stride, 0, &q_rope_tmp1_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0.stride, 0, &q_rope_tmp2_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0.stride, 0, &q_rope_tmp3_h0.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1.stride, 0, &q_rope_src_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1.stride, 0, &q_rope_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1.stride, 0, &q_rope_tmp0_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1.stride, 0, &q_rope_tmp1_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1.stride, 0, &q_rope_tmp2_h1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1.stride, 0, &q_rope_tmp3_h1.shape, DT_BFP16);
  __ppl_tensor_info cos_shared = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared.stride, 0, &cos_shared.shape, DT_BFP16);
  __ppl_tensor_info sin_shared = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared.stride, 0, &sin_shared.shape, DT_BFP16);
  __ppl_tensor_info pe_cur = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur.stride, 0, &pe_cur.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos.stride, 0, &pe_cur_cos.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin.stride, 0, &pe_cur_sin.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg.stride, 0, &pe_cur_neg.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin.stride, 0, &pe_cur_neg_sin.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope.stride, 0, &pe_cur_rope.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile.stride, 0, &wukv_full_tile.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full.stride, 0, &kv_cache_full.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile.stride, 0, &pe_score_tile.shape, DT_BFP16);
  __ppl_tensor_info mask_row = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row.stride, 0, &mask_row.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0.stride, 0, &score_latent_h0.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0.stride, 0, &score_rope_h0.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1.stride, 0, &score_latent_h1.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1.stride, 0, &score_rope_h1.shape, DT_FP32);
  if (tpu_workitem_index() == 0) {
    __ppl_tensor_info COS = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_1 = {.shape = {1, 1, 1, 64} , .stride = cos_shared.stride, .addr = cos_shared.addr + ((0) * cos_shared.stride.c+(0) * cos_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared.stride.c+(0) * cos_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_1.addr, COS.addr, &cos_shared_1.shape, (cos_shared_1.default_stride ? NULL : &cos_shared_1.stride), (COS.default_stride ? NULL : &COS.stride), DT_BFP16);
    __ppl_tensor_info SIN = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_1 = {.shape = {1, 1, 1, 64} , .stride = sin_shared.stride, .addr = sin_shared.addr + ((0) * sin_shared.stride.c+(0) * sin_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared.stride.c+(0) * sin_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_1.addr, SIN.addr, &sin_shared_1.shape, (sin_shared_1.default_stride ? NULL : &sin_shared_1.stride), (SIN.default_stride ? NULL : &SIN.stride), DT_BFP16);
    for (int d = 0; d < 4; ++d) {
      __ppl_tensor_info kv_cur_tile = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile.stride, 0, &kv_cur_tile.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_1 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile.stride, .addr = kv_cur_tile.addr + ((0) * kv_cur_tile.stride.c+(0) * kv_cur_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile.stride.c+(0) * kv_cur_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_1.addr, KV_CUR.addr, &kv_cur_tile_1.shape, (kv_cur_tile_1.default_stride ? NULL : &kv_cur_tile_1.stride), (KV_CUR.default_stride ? NULL : &KV_CUR.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_2 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile.stride, .addr = kv_cur_tile.addr + ((0) * kv_cur_tile.stride.c+(0) * kv_cur_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile.stride.c+(0) * kv_cur_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile.default_stride};
      __ppl_tensor_info KVcache_OUT = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT.addr, kv_cur_tile_2.addr, &KVcache_OUT.shape, (KVcache_OUT.default_stride ? NULL : &KVcache_OUT.stride), (kv_cur_tile_2.default_stride ? NULL : &kv_cur_tile_2.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_1 = {.shape = {1, 1, 1, 64} , .stride = pe_cur.stride, .addr = pe_cur.addr + ((0) * pe_cur.stride.c+(0) * pe_cur.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur.stride.c+(0) * pe_cur.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_1.addr, PE_CUR.addr, &pe_cur_1.shape, (pe_cur_1.default_stride ? NULL : &pe_cur_1.stride), (PE_CUR.default_stride ? NULL : &PE_CUR.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos.addr, pe_cur.addr, cos_shared.addr, &pe_cur_cos.shape, (pe_cur_cos.default_stride ? NULL : &pe_cur_cos.stride), (pe_cur.default_stride ? NULL : &pe_cur.stride), (cos_shared.default_stride ? NULL : &cos_shared.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin.addr, pe_cur.addr, sin_shared.addr, &pe_cur_sin.shape, (pe_cur_sin.default_stride ? NULL : &pe_cur_sin.stride), (pe_cur.default_stride ? NULL : &pe_cur.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg.addr, pe_cur.addr, pe_cur_neg_scalar_DT_BFP16, &pe_cur_neg.shape, (pe_cur_neg.default_stride ? NULL : &pe_cur_neg.stride), (pe_cur.default_stride ? NULL : &pe_cur.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin.addr, pe_cur_neg.addr, sin_shared.addr, &pe_cur_neg_sin.shape, (pe_cur_neg_sin.default_stride ? NULL : &pe_cur_neg_sin.stride), (pe_cur_neg.default_stride ? NULL : &pe_cur_neg.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope.shape.n, .c = pe_cur_rope.shape.c, .h = pe_cur_rope.shape.h, .w = pe_cur_rope.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope.addr, pe_cur_cos.addr, pe_cur_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope.addr + 2, pe_cur_cos.addr + 2, pe_cur_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_1 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope.stride, .addr = pe_cur_rope.addr + ((0) * pe_cur_rope.stride.c+(0) * pe_cur_rope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope.stride.c+(0) * pe_cur_rope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope.default_stride};
    __ppl_tensor_info PEcache_OUT = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT.addr, pe_cur_rope_1.addr, &PEcache_OUT.shape, (PEcache_OUT.default_stride ? NULL : &PEcache_OUT.stride), (pe_cur_rope_1.default_stride ? NULL : &pe_cur_rope_1.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0.addr, q_upper_acc_h0_scalar_f32, &q_upper_acc_h0.shape, (q_upper_acc_h0.default_stride ? NULL : &q_upper_acc_h0.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1.addr, q_upper_acc_h1_scalar_f32, &q_upper_acc_h1.shape, (q_upper_acc_h1.default_stride ? NULL : &q_upper_acc_h1.stride), DT_FP32);
    }
    for (int k = 0; k < 12; ++k) {
      __ppl_tensor_info q_tile = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile.stride, 0, &q_tile.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8.stride, 0, &wuq_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage.stride, 0, &wuq_scale_stage.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile.stride, 0, &wuq_tile.shape, DT_BFP16);
      __ppl_tensor_info Q = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_1 = {.shape = {1, 1, 1, 128} , .stride = q_tile.stride, .addr = q_tile.addr + ((0) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile.default_stride};
      tpu_gdma_cpy_S2L(q_tile_1.addr, Q.addr, &q_tile_1.shape, (q_tile_1.default_stride ? NULL : &q_tile_1.stride), (Q.default_stride ? NULL : &Q.stride), DT_BFP16);
      __ppl_tensor_info WUQ = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((0) * 1536+((k * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((0) * 1536+((k * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_1 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8.stride, .addr = wuq_fp8.addr + ((0) * wuq_fp8.stride.c+(0) * wuq_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8.stride.c+(0) * wuq_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_1.addr, WUQ.addr, &wuq_fp8_1.shape, (wuq_fp8_1.default_stride ? NULL : &wuq_fp8_1.stride), (WUQ.default_stride ? NULL : &WUQ.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar.stride, 0, &wuq_scale_scalar.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((0) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_1 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar.stride, .addr = wuq_scale_scalar.addr + ((0) * wuq_scale_scalar.stride.c+(0) * wuq_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar.stride.c+(0) * wuq_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_1.addr, WUQ_SCALE.addr, &wuq_scale_scalar_1.shape, (wuq_scale_scalar_1.default_stride ? NULL : &wuq_scale_scalar_1.stride), (WUQ_SCALE.default_stride ? NULL : &WUQ_SCALE.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_2 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar.stride, .addr = wuq_scale_scalar.addr + (((0) / 64) * wuq_scale_scalar.stride.c+(0) * wuq_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar.stride.c+(0) * wuq_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_1 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((0) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_1.addr, wuq_scale_scalar_2.addr, &wuq_scale_stage_1.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_3 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_3.stride, 0, &wuq_scale_scalar_3.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_1 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((0) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_4 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_3.stride, .addr = wuq_scale_scalar_3.addr + ((0) * wuq_scale_scalar_3.stride.c+(0) * wuq_scale_scalar_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_3.stride.c+(0) * wuq_scale_scalar_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_3.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_4.addr, WUQ_SCALE_1.addr, &wuq_scale_scalar_4.shape, (wuq_scale_scalar_4.default_stride ? NULL : &wuq_scale_scalar_4.stride), (WUQ_SCALE_1.default_stride ? NULL : &WUQ_SCALE_1.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_5 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_3.stride, .addr = wuq_scale_scalar_3.addr + (((0) / 64) * wuq_scale_scalar_3.stride.c+(0) * wuq_scale_scalar_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_3.stride.c+(0) * wuq_scale_scalar_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_2 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((64) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_2.addr, wuq_scale_scalar_5.addr, &wuq_scale_stage_2.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_6 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_6.stride, 0, &wuq_scale_scalar_6.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_2 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((1) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((1) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_7 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_6.stride, .addr = wuq_scale_scalar_6.addr + ((0) * wuq_scale_scalar_6.stride.c+(0) * wuq_scale_scalar_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_6.stride.c+(0) * wuq_scale_scalar_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_6.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_7.addr, WUQ_SCALE_2.addr, &wuq_scale_scalar_7.shape, (wuq_scale_scalar_7.default_stride ? NULL : &wuq_scale_scalar_7.stride), (WUQ_SCALE_2.default_stride ? NULL : &WUQ_SCALE_2.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_8 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_6.stride, .addr = wuq_scale_scalar_6.addr + (((0) / 64) * wuq_scale_scalar_6.stride.c+(0) * wuq_scale_scalar_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_6.stride.c+(0) * wuq_scale_scalar_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_3 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((128) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_3.addr, wuq_scale_scalar_8.addr, &wuq_scale_stage_3.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_9 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_9.stride, 0, &wuq_scale_scalar_9.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_3 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((1) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((1) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_10 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_9.stride, .addr = wuq_scale_scalar_9.addr + ((0) * wuq_scale_scalar_9.stride.c+(0) * wuq_scale_scalar_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_9.stride.c+(0) * wuq_scale_scalar_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_9.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_10.addr, WUQ_SCALE_3.addr, &wuq_scale_scalar_10.shape, (wuq_scale_scalar_10.default_stride ? NULL : &wuq_scale_scalar_10.stride), (WUQ_SCALE_3.default_stride ? NULL : &WUQ_SCALE_3.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_11 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_9.stride, .addr = wuq_scale_scalar_9.addr + (((0) / 64) * wuq_scale_scalar_9.stride.c+(0) * wuq_scale_scalar_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_9.stride.c+(0) * wuq_scale_scalar_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_4 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((192) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_4.addr, wuq_scale_scalar_11.addr, &wuq_scale_stage_4.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_12 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_12.stride, 0, &wuq_scale_scalar_12.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_4 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((2) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((2) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_13 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_12.stride, .addr = wuq_scale_scalar_12.addr + ((0) * wuq_scale_scalar_12.stride.c+(0) * wuq_scale_scalar_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_12.stride.c+(0) * wuq_scale_scalar_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_12.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_13.addr, WUQ_SCALE_4.addr, &wuq_scale_scalar_13.shape, (wuq_scale_scalar_13.default_stride ? NULL : &wuq_scale_scalar_13.stride), (WUQ_SCALE_4.default_stride ? NULL : &WUQ_SCALE_4.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_14 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_12.stride, .addr = wuq_scale_scalar_12.addr + (((0) / 64) * wuq_scale_scalar_12.stride.c+(0) * wuq_scale_scalar_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_12.stride.c+(0) * wuq_scale_scalar_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_5 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((256) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_5.addr, wuq_scale_scalar_14.addr, &wuq_scale_stage_5.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_15 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_15.stride, 0, &wuq_scale_scalar_15.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_5 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((2) * 12+(k) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((2) * 12+(k) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_16 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_15.stride, .addr = wuq_scale_scalar_15.addr + ((0) * wuq_scale_scalar_15.stride.c+(0) * wuq_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_15.stride.c+(0) * wuq_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_15.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_16.addr, WUQ_SCALE_5.addr, &wuq_scale_scalar_16.shape, (wuq_scale_scalar_16.default_stride ? NULL : &wuq_scale_scalar_16.stride), (WUQ_SCALE_5.default_stride ? NULL : &WUQ_SCALE_5.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_17 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_15.stride, .addr = wuq_scale_scalar_15.addr + (((0) / 64) * wuq_scale_scalar_15.stride.c+(0) * wuq_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_15.stride.c+(0) * wuq_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_6 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((320) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_6.addr, wuq_scale_scalar_17.addr, &wuq_scale_stage_6.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_2 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8.stride, .addr = wuq_fp8.addr + (((0) / 64) * wuq_fp8.stride.c+(0) * wuq_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8.stride.c+(0) * wuq_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_7 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage.stride, .addr = wuq_scale_stage.addr + (((0) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage.stride.c+(0) * wuq_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_1 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile.stride, .addr = wuq_tile.addr + (((0) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_1.addr, wuq_fp8_2.addr, &wuq_tile_1.shape, (wuq_tile_1.default_stride ? NULL : &wuq_tile_1.stride), (wuq_fp8_2.default_stride ? NULL : &wuq_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_7_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_7_block_stride, 0, &wuq_scale_stage_7.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_7_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_1.addr, wuq_tile_1.addr, wuq_scale_stage_7.addr, &wuq_tile_1.shape, (wuq_tile_1.default_stride ? NULL : &wuq_tile_1.stride), (wuq_tile_1.default_stride ? NULL : &wuq_tile_1.stride), &wuq_scale_stage_7_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_2 = {.shape = {1, 1, 1, 128}, .stride = q_tile.stride, .addr = q_tile.addr + (((0) / 64) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_2 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile.stride, .addr = wuq_tile.addr + (((0) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_1 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0.stride, .addr = q_upper_acc_h0.addr + (((0) / 64) * q_upper_acc_h0.stride.c+(0) * q_upper_acc_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0.stride.c+(0) * q_upper_acc_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_1.addr, q_tile_2.addr, wuq_tile_2.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_3 = {.shape = {1, 1, 1, 128}, .stride = q_tile.stride, .addr = q_tile.addr + (((0) / 64) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_3 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile.stride, .addr = wuq_tile.addr + (((192) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile.stride.c+(0) * wuq_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_1 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1.stride, .addr = q_upper_acc_h1.addr + (((0) / 64) * q_upper_acc_h1.stride.c+(0) * q_upper_acc_h1.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1.stride.c+(0) * q_upper_acc_h1.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_1.addr, q_tile_3.addr, wuq_tile_3.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_2 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0.stride, .addr = q_upper_acc_h0.addr + ((0) * q_upper_acc_h0.stride.c+(0) * q_upper_acc_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0.stride.c+(0) * q_upper_acc_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0.default_stride};
    __ppl_tensor_info q_upper_h0_1 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0.stride, .addr = q_upper_h0.addr + ((0) * q_upper_h0.stride.c+(0) * q_upper_h0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0.stride.c+(0) * q_upper_h0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0.default_stride};
    tpu_bdc_cast(q_upper_h0_1.addr, q_upper_acc_h0_2.addr, &q_upper_h0_1.shape, (q_upper_h0_1.default_stride ? NULL : &q_upper_h0_1.stride), (q_upper_acc_h0_2.default_stride ? NULL : &q_upper_acc_h0_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_2 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1.stride, .addr = q_upper_acc_h1.addr + ((0) * q_upper_acc_h1.stride.c+(0) * q_upper_acc_h1.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1.stride.c+(0) * q_upper_acc_h1.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1.default_stride};
    __ppl_tensor_info q_upper_h1_1 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1.stride, .addr = q_upper_h1.addr + ((0) * q_upper_h1.stride.c+(0) * q_upper_h1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1.stride.c+(0) * q_upper_h1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1.default_stride};
    tpu_bdc_cast(q_upper_h1_1.addr, q_upper_acc_h1_2.addr, &q_upper_h1_1.shape, (q_upper_h1_1.default_stride ? NULL : &q_upper_h1_1.stride), (q_upper_acc_h1_2.default_stride ? NULL : &q_upper_acc_h1_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_1 = 0; d_1 < 4; ++d_1) {
      __ppl_tensor_info wukv_fp8 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8.stride, 0, &wukv_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage.stride, 0, &wukv_scale_stage.shape, DT_BFP16);
      __ppl_tensor_info WUKV = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((0) * 512+((d_1 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((0) * 512+((d_1 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_1 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8.stride, .addr = wukv_fp8.addr + ((0) * wukv_fp8.stride.c+(0) * wukv_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8.stride.c+(0) * wukv_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_1.addr, WUKV.addr, &wukv_fp8_1.shape, (wukv_fp8_1.default_stride ? NULL : &wukv_fp8_1.stride), (WUKV.default_stride ? NULL : &WUKV.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar.stride, 0, &wukv_scale_scalar.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((0) * 4+(d_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 4+(d_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_1 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar.stride, .addr = wukv_scale_scalar.addr + ((0) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_1.addr, WUKV_SCALE.addr, &wukv_scale_scalar_1.shape, (wukv_scale_scalar_1.default_stride ? NULL : &wukv_scale_scalar_1.stride), (WUKV_SCALE.default_stride ? NULL : &WUKV_SCALE.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_2 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar.stride, .addr = wukv_scale_scalar.addr + (((0) / 64) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_1 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage.stride, .addr = wukv_scale_stage.addr + (((0) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_1_chunk = wukv_scale_stage_1;
      wukv_scale_stage_1_chunk.shape.c = 64;
      wukv_scale_stage_1_chunk.addr = wukv_scale_stage_1.addr + ((0) * wukv_scale_stage_1.stride.c) * 2;
      wukv_scale_stage_1_chunk.offset = wukv_scale_stage_1.offset + ((0) * wukv_scale_stage_1.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_1_chunk.addr, wukv_scale_scalar_2.addr, &wukv_scale_stage_1_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_1_chunk_1 = wukv_scale_stage_1;
      wukv_scale_stage_1_chunk_1.shape.c = 64;
      wukv_scale_stage_1_chunk_1.addr = wukv_scale_stage_1.addr + ((1) * wukv_scale_stage_1.stride.c) * 2;
      wukv_scale_stage_1_chunk_1.offset = wukv_scale_stage_1.offset + ((1) * wukv_scale_stage_1.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_1_chunk_1.addr, wukv_scale_scalar_2.addr, &wukv_scale_stage_1_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_1 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((1) * 4+(d_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((1) * 4+(d_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_3 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar.stride, .addr = wukv_scale_scalar.addr + ((0) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_3.addr, WUKV_SCALE_1.addr, &wukv_scale_scalar_3.shape, (wukv_scale_scalar_3.default_stride ? NULL : &wukv_scale_scalar_3.stride), (WUKV_SCALE_1.default_stride ? NULL : &WUKV_SCALE_1.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_4 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar.stride, .addr = wukv_scale_scalar.addr + (((0) / 64) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar.stride.c+(0) * wukv_scale_scalar.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_2 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage.stride, .addr = wukv_scale_stage.addr + (((128) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_2_chunk = wukv_scale_stage_2;
      wukv_scale_stage_2_chunk.shape.c = 64;
      wukv_scale_stage_2_chunk.addr = wukv_scale_stage_2.addr + ((0) * wukv_scale_stage_2.stride.c) * 2;
      wukv_scale_stage_2_chunk.offset = wukv_scale_stage_2.offset + ((0) * wukv_scale_stage_2.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_2_chunk.addr, wukv_scale_scalar_4.addr, &wukv_scale_stage_2_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_2_chunk_1 = wukv_scale_stage_2;
      wukv_scale_stage_2_chunk_1.shape.c = 64;
      wukv_scale_stage_2_chunk_1.addr = wukv_scale_stage_2.addr + ((1) * wukv_scale_stage_2.stride.c) * 2;
      wukv_scale_stage_2_chunk_1.offset = wukv_scale_stage_2.offset + ((1) * wukv_scale_stage_2.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_2_chunk_1.addr, wukv_scale_scalar_4.addr, &wukv_scale_stage_2_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_5 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_5.stride, 0, &wukv_scale_scalar_5.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_2 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((2) * 4+(d_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((2) * 4+(d_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_6 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_5.stride, .addr = wukv_scale_scalar_5.addr + ((0) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_5.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_6.addr, WUKV_SCALE_2.addr, &wukv_scale_scalar_6.shape, (wukv_scale_scalar_6.default_stride ? NULL : &wukv_scale_scalar_6.stride), (WUKV_SCALE_2.default_stride ? NULL : &WUKV_SCALE_2.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_7 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_5.stride, .addr = wukv_scale_scalar_5.addr + (((0) / 64) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_3 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage.stride, .addr = wukv_scale_stage.addr + (((256) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_3_chunk = wukv_scale_stage_3;
      wukv_scale_stage_3_chunk.shape.c = 64;
      wukv_scale_stage_3_chunk.addr = wukv_scale_stage_3.addr + ((0) * wukv_scale_stage_3.stride.c) * 2;
      wukv_scale_stage_3_chunk.offset = wukv_scale_stage_3.offset + ((0) * wukv_scale_stage_3.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_3_chunk.addr, wukv_scale_scalar_7.addr, &wukv_scale_stage_3_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_3_chunk_1 = wukv_scale_stage_3;
      wukv_scale_stage_3_chunk_1.shape.c = 64;
      wukv_scale_stage_3_chunk_1.addr = wukv_scale_stage_3.addr + ((1) * wukv_scale_stage_3.stride.c) * 2;
      wukv_scale_stage_3_chunk_1.offset = wukv_scale_stage_3.offset + ((1) * wukv_scale_stage_3.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_3_chunk_1.addr, wukv_scale_scalar_7.addr, &wukv_scale_stage_3_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_3 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((3) * 4+(d_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((3) * 4+(d_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_8 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_5.stride, .addr = wukv_scale_scalar_5.addr + ((0) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_5.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_8.addr, WUKV_SCALE_3.addr, &wukv_scale_scalar_8.shape, (wukv_scale_scalar_8.default_stride ? NULL : &wukv_scale_scalar_8.stride), (WUKV_SCALE_3.default_stride ? NULL : &WUKV_SCALE_3.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_9 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_5.stride, .addr = wukv_scale_scalar_5.addr + (((0) / 64) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_5.stride.c+(0) * wukv_scale_scalar_5.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_4 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage.stride, .addr = wukv_scale_stage.addr + (((384) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_4_chunk = wukv_scale_stage_4;
      wukv_scale_stage_4_chunk.shape.c = 64;
      wukv_scale_stage_4_chunk.addr = wukv_scale_stage_4.addr + ((0) * wukv_scale_stage_4.stride.c) * 2;
      wukv_scale_stage_4_chunk.offset = wukv_scale_stage_4.offset + ((0) * wukv_scale_stage_4.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_4_chunk.addr, wukv_scale_scalar_9.addr, &wukv_scale_stage_4_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_4_chunk_1 = wukv_scale_stage_4;
      wukv_scale_stage_4_chunk_1.shape.c = 64;
      wukv_scale_stage_4_chunk_1.addr = wukv_scale_stage_4.addr + ((1) * wukv_scale_stage_4.stride.c) * 2;
      wukv_scale_stage_4_chunk_1.offset = wukv_scale_stage_4.offset + ((1) * wukv_scale_stage_4.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_4_chunk_1.addr, wukv_scale_scalar_9.addr, &wukv_scale_stage_4_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_2 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8.stride, .addr = wukv_fp8.addr + (((0) / 64) * wukv_fp8.stride.c+(0) * wukv_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8.stride.c+(0) * wukv_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_5 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage.stride, .addr = wukv_scale_stage.addr + (((0) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage.stride.c+(0) * wukv_scale_stage.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_1 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile.stride, .addr = wukv_full_tile.addr + (((0) / 64) * wukv_full_tile.stride.c+((d_1 * 128)) * wukv_full_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile.stride.c+((d_1 * 128)) * wukv_full_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_1.addr, wukv_fp8_2.addr, &wukv_full_tile_1.shape, (wukv_full_tile_1.default_stride ? NULL : &wukv_full_tile_1.stride), (wukv_fp8_2.default_stride ? NULL : &wukv_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_5_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_5_block_stride, 0, &wukv_scale_stage_5.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_5_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_1.addr, wukv_full_tile_1.addr, wukv_scale_stage_5.addr, &wukv_full_tile_1.shape, (wukv_full_tile_1.default_stride ? NULL : &wukv_full_tile_1.stride), (wukv_full_tile_1.default_stride ? NULL : &wukv_full_tile_1.stride), &wukv_scale_stage_5_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_1 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_1 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full.stride, .addr = kv_cache_full.addr + ((0) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_1.addr, KVcache_OUT_1.addr, &kv_cache_full_1.shape, (kv_cache_full_1.default_stride ? NULL : &kv_cache_full_1.stride), (KVcache_OUT_1.default_stride ? NULL : &KVcache_OUT_1.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_1 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_1 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile.stride, .addr = pe_score_tile.addr + ((0) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_1.addr, PEcache_OUT_1.addr, &pe_score_tile_1.shape, (pe_score_tile_1.default_stride ? NULL : &pe_score_tile_1.stride), (PEcache_OUT_1.default_stride ? NULL : &PEcache_OUT_1.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_1 = {.shape = {1, 1, 1, 192} , .stride = mask_row.stride, .addr = mask_row.addr + ((0) * mask_row.stride.c+(0) * mask_row.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row.stride.c+(0) * mask_row.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row.default_stride};
    tpu_gdma_cpy_S2L(mask_row_1.addr, DECODE_MASK.addr, &mask_row_1.shape, (mask_row_1.default_stride ? NULL : &mask_row_1.stride), (DECODE_MASK.default_stride ? NULL : &DECODE_MASK.stride), DT_FP32);
    {
    scalar_t score_latent_h0_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0.addr, score_latent_h0_scalar_f32, &score_latent_h0.shape, (score_latent_h0.default_stride ? NULL : &score_latent_h0.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0.addr, score_rope_h0_scalar_f32, &score_rope_h0.shape, (score_rope_h0.default_stride ? NULL : &score_rope_h0.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1.addr, score_latent_h1_scalar_f32, &score_latent_h1.shape, (score_latent_h1.default_stride ? NULL : &score_latent_h1.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1.addr, score_rope_h1_scalar_f32, &score_rope_h1.shape, (score_rope_h1.default_stride ? NULL : &score_rope_h1.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_2 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0.stride, .addr = q_upper_h0.addr + ((0) * q_upper_h0.stride.c+(128) * q_upper_h0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0.stride.c+(128) * q_upper_h0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0.default_stride};
    __ppl_tensor_info q_rope_src_h0_1 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0.stride, .addr = q_rope_src_h0.addr + ((0) * q_rope_src_h0.stride.c+(0) * q_rope_src_h0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0.stride.c+(0) * q_rope_src_h0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_1.addr, q_upper_h0_2.addr, &q_rope_src_h0_1.shape, (q_rope_src_h0_1.default_stride ? NULL : &q_rope_src_h0_1.stride), (q_upper_h0_2.default_stride ? NULL : &q_upper_h0_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0.addr, q_rope_src_h0.addr, cos_shared.addr, &q_rope_tmp0_h0.shape, (q_rope_tmp0_h0.default_stride ? NULL : &q_rope_tmp0_h0.stride), (q_rope_src_h0.default_stride ? NULL : &q_rope_src_h0.stride), (cos_shared.default_stride ? NULL : &cos_shared.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0.addr, q_rope_src_h0.addr, sin_shared.addr, &q_rope_tmp1_h0.shape, (q_rope_tmp1_h0.default_stride ? NULL : &q_rope_tmp1_h0.stride), (q_rope_src_h0.default_stride ? NULL : &q_rope_src_h0.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0.addr, q_rope_src_h0.addr, q_rope_tmp2_h0_scalar_DT_BFP16, &q_rope_tmp2_h0.shape, (q_rope_tmp2_h0.default_stride ? NULL : &q_rope_tmp2_h0.stride), (q_rope_src_h0.default_stride ? NULL : &q_rope_src_h0.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0.addr, q_rope_tmp2_h0.addr, sin_shared.addr, &q_rope_tmp3_h0.shape, (q_rope_tmp3_h0.default_stride ? NULL : &q_rope_tmp3_h0.stride), (q_rope_tmp2_h0.default_stride ? NULL : &q_rope_tmp2_h0.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0.shape.n, .c = q_rope_h0.shape.c, .h = q_rope_h0.shape.h, .w = q_rope_h0.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0.addr, q_rope_tmp0_h0.addr, q_rope_tmp3_h0.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0.addr + 2, q_rope_tmp0_h0.addr + 2, q_rope_tmp1_h0.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_2 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1.stride, .addr = q_upper_h1.addr + ((0) * q_upper_h1.stride.c+(128) * q_upper_h1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1.stride.c+(128) * q_upper_h1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1.default_stride};
    __ppl_tensor_info q_rope_src_h1_1 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1.stride, .addr = q_rope_src_h1.addr + ((0) * q_rope_src_h1.stride.c+(0) * q_rope_src_h1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1.stride.c+(0) * q_rope_src_h1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_1.addr, q_upper_h1_2.addr, &q_rope_src_h1_1.shape, (q_rope_src_h1_1.default_stride ? NULL : &q_rope_src_h1_1.stride), (q_upper_h1_2.default_stride ? NULL : &q_upper_h1_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1.addr, q_rope_src_h1.addr, cos_shared.addr, &q_rope_tmp0_h1.shape, (q_rope_tmp0_h1.default_stride ? NULL : &q_rope_tmp0_h1.stride), (q_rope_src_h1.default_stride ? NULL : &q_rope_src_h1.stride), (cos_shared.default_stride ? NULL : &cos_shared.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1.addr, q_rope_src_h1.addr, sin_shared.addr, &q_rope_tmp1_h1.shape, (q_rope_tmp1_h1.default_stride ? NULL : &q_rope_tmp1_h1.stride), (q_rope_src_h1.default_stride ? NULL : &q_rope_src_h1.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1.addr, q_rope_src_h1.addr, q_rope_tmp2_h1_scalar_DT_BFP16, &q_rope_tmp2_h1.shape, (q_rope_tmp2_h1.default_stride ? NULL : &q_rope_tmp2_h1.stride), (q_rope_src_h1.default_stride ? NULL : &q_rope_src_h1.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1.addr, q_rope_tmp2_h1.addr, sin_shared.addr, &q_rope_tmp3_h1.shape, (q_rope_tmp3_h1.default_stride ? NULL : &q_rope_tmp3_h1.stride), (q_rope_tmp2_h1.default_stride ? NULL : &q_rope_tmp2_h1.stride), (sin_shared.default_stride ? NULL : &sin_shared.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1.shape.n, .c = q_rope_h1.shape.c, .h = q_rope_h1.shape.h, .w = q_rope_h1.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1.addr, q_rope_tmp0_h1.addr, q_rope_tmp3_h1.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1.addr + 2, q_rope_tmp0_h1.addr + 2, q_rope_tmp1_h1.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head.stride, 0, &q_nope_head.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc.stride, 0, &q_wukv_acc.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head.stride, 0, &q_wukv_head.shape, DT_BFP16);
    __ppl_tensor_info score_tile = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile.stride, 0, &score_tile.shape, DT_FP32);
    __ppl_tensor_info prob_tile = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile.stride, 0, &prob_tile.shape, DT_BFP16);
    __ppl_tensor_info scores_max = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max.stride, 0, &scores_max.shape, DT_FP32);
    __ppl_tensor_info scores_sum = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum.stride, 0, &scores_sum.shape, DT_FP32);
    __ppl_tensor_info work_score0 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0.stride, 0, &work_score0.shape, DT_FP32);
    __ppl_tensor_info work_score1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1.stride, 0, &work_score1.shape, DT_FP32);
    __ppl_tensor_info coeff = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff.stride, 0, &coeff.shape, DT_FP32);
    __ppl_tensor_info table = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table.stride, 0, &table.shape, DT_FP32);
    __ppl_tensor_info ctx_acc = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc.stride, 0, &ctx_acc.shape, DT_FP32);
    __ppl_tensor_info ctx_head = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head.stride, 0, &ctx_head.shape, DT_BFP16);
    __ppl_tensor_info out_part = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part.stride, 0, &out_part.shape, DT_FP32);
    __ppl_tensor_info out_cast = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast.stride, 0, &out_cast.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_3 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0.stride, .addr = q_upper_h0.addr + ((0) * q_upper_h0.stride.c+(0) * q_upper_h0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0.stride.c+(0) * q_upper_h0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0.default_stride};
    __ppl_tensor_info q_nope_head_1 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head.stride, .addr = q_nope_head.addr + ((0) * q_nope_head.stride.c+(0) * q_nope_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head.stride.c+(0) * q_nope_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head.default_stride};
    tpu_bdc_cpy(q_nope_head_1.addr, q_upper_h0_3.addr, &q_nope_head_1.shape, (q_nope_head_1.default_stride ? NULL : &q_nope_head_1.stride), (q_upper_h0_3.default_stride ? NULL : &q_upper_h0_3.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc.addr, q_wukv_acc_scalar_f32, &q_wukv_acc.shape, (q_wukv_acc.default_stride ? NULL : &q_wukv_acc.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_2 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head.stride, .addr = q_nope_head.addr + (((0) / 64) * q_nope_head.stride.c+(0) * q_nope_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head.stride.c+(0) * q_nope_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_2 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile.stride, .addr = wukv_full_tile.addr + (((0) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_1 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc.stride, .addr = q_wukv_acc.addr + (((0) / 64) * q_wukv_acc.stride.c+(0) * q_wukv_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc.stride.c+(0) * q_wukv_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_1.addr, q_nope_head_2.addr, wukv_full_tile_2.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_2 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc.stride, .addr = q_wukv_acc.addr + ((0) * q_wukv_acc.stride.c+(0) * q_wukv_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc.stride.c+(0) * q_wukv_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc.default_stride};
    __ppl_tensor_info q_wukv_head_1 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head.stride, .addr = q_wukv_head.addr + ((0) * q_wukv_head.stride.c+(0) * q_wukv_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head.stride.c+(0) * q_wukv_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head.default_stride};
    tpu_bdc_cast(q_wukv_head_1.addr, q_wukv_acc_2.addr, &q_wukv_head_1.shape, (q_wukv_head_1.default_stride ? NULL : &q_wukv_head_1.stride), (q_wukv_acc_2.default_stride ? NULL : &q_wukv_acc_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_2 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head.stride, .addr = q_wukv_head.addr + (((0) / 64) * q_wukv_head.stride.c+(0) * q_wukv_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head.stride.c+(0) * q_wukv_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_2 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full.stride, .addr = kv_cache_full.addr + (((0) / 64) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_1 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0.stride, .addr = score_latent_h0.addr + (((0) / 64) * score_latent_h0.stride.c+(0) * score_latent_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0.stride.c+(0) * score_latent_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_1.addr, q_wukv_head_2.addr, kv_cache_full_2.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_1 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0.stride, .addr = q_rope_h0.addr + (((0) / 64) * q_rope_h0.stride.c+(0) * q_rope_h0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0.stride.c+(0) * q_rope_h0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_2 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile.stride, .addr = pe_score_tile.addr + (((0) / 64) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_1 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0.stride, .addr = score_rope_h0.addr + (((0) / 64) * score_rope_h0.stride.c+(0) * score_rope_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0.stride.c+(0) * score_rope_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_1.addr, q_rope_h0_1.addr, pe_score_tile_2.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile.addr, score_latent_h0.addr, score_rope_h0.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_latent_h0.default_stride ? NULL : &score_latent_h0.stride), (score_rope_h0.default_stride ? NULL : &score_rope_h0.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile.addr, score_tile.addr, mask_row.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), (mask_row.default_stride ? NULL : &mask_row.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile.addr, score_tile.addr, (scalar_t){.f32 = 0.0721688}, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), DT_FP32);
    {
    scalar_t scores_max_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max.stride, 0, &tmp_buffer_max.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile.shape.n, score_tile.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile.shape.n, score_tile.shape.c, 1, score_tile.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile.shape.w) {
          dim4 fill_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w - score_tile.shape.w};
          int elem_size = 4;
          int offset = score_tile.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_stride;
    tpu_aligned_stride(&scores_max_stride, 0, &scores_max.shape, DT_FP32);
    scores_max_stride.w = 0;
    tpu_bdc_fp_sub( score_tile.addr, score_tile.addr, scores_max.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), &scores_max_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff.addr);
    tpu_bdc_load_fp32_exp_table(table.addr);
    tpu_bdc_fp32_exp(score_tile.addr, score_tile.addr, work_score0.addr, work_score1.addr, coeff.addr, table.addr, &score_tile.shape);
    __ppl_tensor_info tmp_buffer_sum = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum.stride, 0, &tmp_buffer_sum.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile.shape.n, score_tile.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile.shape.n, score_tile.shape.c, 1, score_tile.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile.shape.w) {
          dim4 fill_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w - score_tile.shape.w};
          int elem_size = 4;
          int offset = score_tile.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_1 = {.shape = {1, 1, 1, 192} , .stride = score_tile.stride, .addr = score_tile.addr + ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile.default_stride};
    __ppl_tensor_info prob_tile_1 = {.shape = {1, 1, 1, 192} , .stride = prob_tile.stride, .addr = prob_tile.addr + ((0) * prob_tile.stride.c+(0) * prob_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile.stride.c+(0) * prob_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile.default_stride};
    tpu_bdc_cast(prob_tile_1.addr, score_tile_1.addr, &prob_tile_1.shape, (prob_tile_1.default_stride ? NULL : &prob_tile_1.stride), (score_tile_1.default_stride ? NULL : &score_tile_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc.addr, ctx_acc_scalar_f32, &ctx_acc.shape, (ctx_acc.default_stride ? NULL : &ctx_acc.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc.addr, prob_tile.addr, kv_cache_full.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_stride;
    tpu_aligned_stride(&scores_sum_stride, 0, &scores_sum.shape, DT_FP32);
    scores_sum_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc.addr, ctx_acc.addr, scores_sum.addr, &ctx_acc.shape, (ctx_acc.default_stride ? NULL : &ctx_acc.stride), (ctx_acc.default_stride ? NULL : &ctx_acc.stride), &scores_sum_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_1 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc.stride, .addr = ctx_acc.addr + ((0) * ctx_acc.stride.c+(0) * ctx_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc.stride.c+(0) * ctx_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc.default_stride};
    __ppl_tensor_info ctx_head_1 = {.shape = {1, 1, 1, 512} , .stride = ctx_head.stride, .addr = ctx_head.addr + ((0) * ctx_head.stride.c+(0) * ctx_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head.stride.c+(0) * ctx_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head.default_stride};
    tpu_bdc_cast(ctx_head_1.addr, ctx_acc_1.addr, &ctx_head_1.shape, (ctx_head_1.default_stride ? NULL : &ctx_head_1.stride), (ctx_acc_1.default_stride ? NULL : &ctx_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part.addr, out_part_scalar_f32, &out_part.shape, (out_part.default_stride ? NULL : &out_part.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_2 = {.shape = {1, 1, 1, 512}, .stride = ctx_head.stride, .addr = ctx_head.addr + (((0) / 64) * ctx_head.stride.c+(0) * ctx_head.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head.stride.c+(0) * ctx_head.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_3 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile.stride, .addr = wukv_full_tile.addr + (((128) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_1 = {.shape = {1, 1, 1, 128}, .stride = out_part.stride, .addr = out_part.addr + (((0) / 64) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_1.addr, ctx_head_2.addr, wukv_full_tile_3.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_2 = {.shape = {1, 1, 1, 128} , .stride = out_part.stride, .addr = out_part.addr + ((0) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part.default_stride};
    __ppl_tensor_info out_cast_1 = {.shape = {1, 1, 1, 128} , .stride = out_cast.stride, .addr = out_cast.addr + ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast.default_stride};
    tpu_bdc_cast(out_cast_1.addr, out_part_2.addr, &out_cast_1.shape, (out_cast_1.default_stride ? NULL : &out_cast_1.stride), (out_part_2.default_stride ? NULL : &out_part_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_2 = {.shape = {1, 1, 1, 128} , .stride = out_cast.stride, .addr = out_cast.addr + ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast.default_stride};
    __ppl_tensor_info OUT = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT.addr, out_cast_2.addr, &OUT.shape, (OUT.default_stride ? NULL : &OUT.stride), (out_cast_2.default_stride ? NULL : &out_cast_2.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_3 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_3.stride, 0, &q_nope_head_3.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_3 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_3.stride, 0, &q_wukv_acc_3.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_3 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_3.stride, 0, &q_wukv_head_3.shape, DT_BFP16);
    __ppl_tensor_info score_tile_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_2.stride, 0, &score_tile_2.shape, DT_FP32);
    __ppl_tensor_info prob_tile_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_2.stride, 0, &prob_tile_2.shape, DT_BFP16);
    __ppl_tensor_info scores_max_1 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_1.stride, 0, &scores_max_1.shape, DT_FP32);
    __ppl_tensor_info scores_sum_1 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_1.stride, 0, &scores_sum_1.shape, DT_FP32);
    __ppl_tensor_info work_score0_1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_1.stride, 0, &work_score0_1.shape, DT_FP32);
    __ppl_tensor_info work_score1_1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_1.stride, 0, &work_score1_1.shape, DT_FP32);
    __ppl_tensor_info coeff_1 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_1.stride, 0, &coeff_1.shape, DT_FP32);
    __ppl_tensor_info table_1 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_1.stride, 0, &table_1.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_2 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_2.stride, 0, &ctx_acc_2.shape, DT_FP32);
    __ppl_tensor_info ctx_head_3 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_3.stride, 0, &ctx_head_3.shape, DT_BFP16);
    __ppl_tensor_info out_part_3 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_3.stride, 0, &out_part_3.shape, DT_FP32);
    __ppl_tensor_info out_cast_3 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_3.stride, 0, &out_cast_3.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_3 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1.stride, .addr = q_upper_h1.addr + ((0) * q_upper_h1.stride.c+(0) * q_upper_h1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1.stride.c+(0) * q_upper_h1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1.default_stride};
    __ppl_tensor_info q_nope_head_4 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_3.stride, .addr = q_nope_head_3.addr + ((0) * q_nope_head_3.stride.c+(0) * q_nope_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_3.stride.c+(0) * q_nope_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_3.default_stride};
    tpu_bdc_cpy(q_nope_head_4.addr, q_upper_h1_3.addr, &q_nope_head_4.shape, (q_nope_head_4.default_stride ? NULL : &q_nope_head_4.stride), (q_upper_h1_3.default_stride ? NULL : &q_upper_h1_3.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_3.addr, q_wukv_acc_3_scalar_f32, &q_wukv_acc_3.shape, (q_wukv_acc_3.default_stride ? NULL : &q_wukv_acc_3.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_5 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_3.stride, .addr = q_nope_head_3.addr + (((0) / 64) * q_nope_head_3.stride.c+(0) * q_nope_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_3.stride.c+(0) * q_nope_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_4 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile.stride, .addr = wukv_full_tile.addr + (((256) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_4 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_3.stride, .addr = q_wukv_acc_3.addr + (((0) / 64) * q_wukv_acc_3.stride.c+(0) * q_wukv_acc_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_3.stride.c+(0) * q_wukv_acc_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_4.addr, q_nope_head_5.addr, wukv_full_tile_4.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_5 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_3.stride, .addr = q_wukv_acc_3.addr + ((0) * q_wukv_acc_3.stride.c+(0) * q_wukv_acc_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_3.stride.c+(0) * q_wukv_acc_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_3.default_stride};
    __ppl_tensor_info q_wukv_head_4 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_3.stride, .addr = q_wukv_head_3.addr + ((0) * q_wukv_head_3.stride.c+(0) * q_wukv_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_3.stride.c+(0) * q_wukv_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_3.default_stride};
    tpu_bdc_cast(q_wukv_head_4.addr, q_wukv_acc_5.addr, &q_wukv_head_4.shape, (q_wukv_head_4.default_stride ? NULL : &q_wukv_head_4.stride), (q_wukv_acc_5.default_stride ? NULL : &q_wukv_acc_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_5 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_3.stride, .addr = q_wukv_head_3.addr + (((0) / 64) * q_wukv_head_3.stride.c+(0) * q_wukv_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_3.stride.c+(0) * q_wukv_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_3 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full.stride, .addr = kv_cache_full.addr + (((0) / 64) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full.stride.c+(0) * kv_cache_full.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_2 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0.stride, .addr = score_latent_h0.addr + (((0) / 64) * score_latent_h0.stride.c+(0) * score_latent_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0.stride.c+(0) * score_latent_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_2.addr, q_wukv_head_5.addr, kv_cache_full_3.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_1 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1.stride, .addr = q_rope_h1.addr + (((0) / 64) * q_rope_h1.stride.c+(0) * q_rope_h1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1.stride.c+(0) * q_rope_h1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_3 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile.stride, .addr = pe_score_tile.addr + (((0) / 64) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_2 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0.stride, .addr = score_rope_h0.addr + (((0) / 64) * score_rope_h0.stride.c+(0) * score_rope_h0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0.stride.c+(0) * score_rope_h0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_2.addr, q_rope_h1_1.addr, pe_score_tile_3.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_2.addr, score_latent_h0.addr, score_rope_h0.addr, &score_tile_2.shape, (score_tile_2.default_stride ? NULL : &score_tile_2.stride), (score_latent_h0.default_stride ? NULL : &score_latent_h0.stride), (score_rope_h0.default_stride ? NULL : &score_rope_h0.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_2.addr, score_tile_2.addr, mask_row.addr, &score_tile_2.shape, (score_tile_2.default_stride ? NULL : &score_tile_2.stride), (score_tile_2.default_stride ? NULL : &score_tile_2.stride), (mask_row.default_stride ? NULL : &mask_row.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_2.addr, score_tile_2.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_2.shape, (score_tile_2.default_stride ? NULL : &score_tile_2.stride), (score_tile_2.default_stride ? NULL : &score_tile_2.stride), DT_FP32);
    {
    scalar_t scores_max_1_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_1.addr, scores_max_1_scalar_f32, &scores_max_1.shape, (scores_max_1.default_stride ? NULL : &scores_max_1.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_1 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_1.stride, 0, &tmp_buffer_max_1.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_2.shape.n, score_tile_2.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_2.shape.n, score_tile_2.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_2.shape.n, score_tile_2.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_2.shape.n, score_tile_2.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_2.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, score_tile_2.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_2.shape.w) {
          dim4 fill_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, align_w - score_tile_2.shape.w};
          int elem_size = 4;
          int offset = score_tile_2.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_2.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_2.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_1_stride;
    tpu_aligned_stride(&scores_max_1_stride, 0, &scores_max_1.shape, DT_FP32);
    scores_max_1_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_2.addr, score_tile_2.addr, scores_max_1.addr, &score_tile_2.shape, (score_tile_2.default_stride ? NULL : &score_tile_2.stride), (score_tile_2.default_stride ? NULL : &score_tile_2.stride), &scores_max_1_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_1.addr);
    tpu_bdc_load_fp32_exp_table(table_1.addr);
    tpu_bdc_fp32_exp(score_tile_2.addr, score_tile_2.addr, work_score0_1.addr, work_score1_1.addr, coeff_1.addr, table_1.addr, &score_tile_2.shape);
    __ppl_tensor_info tmp_buffer_sum_1 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_1.stride, 0, &tmp_buffer_sum_1.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_2.shape.n, score_tile_2.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_2.shape.n, score_tile_2.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_2.shape.n, score_tile_2.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_2.shape.n, score_tile_2.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_2.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, score_tile_2.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_2.shape.w) {
          dim4 fill_shape = {score_tile_2.shape.n, score_tile_2.shape.c, 1, align_w - score_tile_2.shape.w};
          int elem_size = 4;
          int offset = score_tile_2.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_2.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_2.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_1.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_3 = {.shape = {1, 1, 1, 192} , .stride = score_tile_2.stride, .addr = score_tile_2.addr + ((0) * score_tile_2.stride.c+(0) * score_tile_2.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_2.stride.c+(0) * score_tile_2.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_2.default_stride};
    __ppl_tensor_info prob_tile_3 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_2.stride, .addr = prob_tile_2.addr + ((0) * prob_tile_2.stride.c+(0) * prob_tile_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_2.stride.c+(0) * prob_tile_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_2.default_stride};
    tpu_bdc_cast(prob_tile_3.addr, score_tile_3.addr, &prob_tile_3.shape, (prob_tile_3.default_stride ? NULL : &prob_tile_3.stride), (score_tile_3.default_stride ? NULL : &score_tile_3.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_2_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_2.addr, ctx_acc_2_scalar_f32, &ctx_acc_2.shape, (ctx_acc_2.default_stride ? NULL : &ctx_acc_2.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_2.addr, prob_tile_2.addr, kv_cache_full.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_1_stride;
    tpu_aligned_stride(&scores_sum_1_stride, 0, &scores_sum_1.shape, DT_FP32);
    scores_sum_1_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_2.addr, ctx_acc_2.addr, scores_sum_1.addr, &ctx_acc_2.shape, (ctx_acc_2.default_stride ? NULL : &ctx_acc_2.stride), (ctx_acc_2.default_stride ? NULL : &ctx_acc_2.stride), &scores_sum_1_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_3 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_2.stride, .addr = ctx_acc_2.addr + ((0) * ctx_acc_2.stride.c+(0) * ctx_acc_2.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_2.stride.c+(0) * ctx_acc_2.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_2.default_stride};
    __ppl_tensor_info ctx_head_4 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_3.stride, .addr = ctx_head_3.addr + ((0) * ctx_head_3.stride.c+(0) * ctx_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_3.stride.c+(0) * ctx_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_3.default_stride};
    tpu_bdc_cast(ctx_head_4.addr, ctx_acc_3.addr, &ctx_head_4.shape, (ctx_head_4.default_stride ? NULL : &ctx_head_4.stride), (ctx_acc_3.default_stride ? NULL : &ctx_acc_3.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_3.addr, out_part_3_scalar_f32, &out_part_3.shape, (out_part_3.default_stride ? NULL : &out_part_3.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_5 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_3.stride, .addr = ctx_head_3.addr + (((0) / 64) * ctx_head_3.stride.c+(0) * ctx_head_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_3.stride.c+(0) * ctx_head_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_5 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile.stride, .addr = wukv_full_tile.addr + (((384) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile.stride.c+(0) * wukv_full_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_4 = {.shape = {1, 1, 1, 128}, .stride = out_part_3.stride, .addr = out_part_3.addr + (((0) / 64) * out_part_3.stride.c+(0) * out_part_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_3.stride.c+(0) * out_part_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_4.addr, ctx_head_5.addr, wukv_full_tile_5.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_5 = {.shape = {1, 1, 1, 128} , .stride = out_part_3.stride, .addr = out_part_3.addr + ((0) * out_part_3.stride.c+(0) * out_part_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_3.stride.c+(0) * out_part_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_3.default_stride};
    __ppl_tensor_info out_cast_4 = {.shape = {1, 1, 1, 128} , .stride = out_cast_3.stride, .addr = out_cast_3.addr + ((0) * out_cast_3.stride.c+(0) * out_cast_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_3.stride.c+(0) * out_cast_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_3.default_stride};
    tpu_bdc_cast(out_cast_4.addr, out_part_5.addr, &out_cast_4.shape, (out_cast_4.default_stride ? NULL : &out_cast_4.stride), (out_part_5.default_stride ? NULL : &out_part_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_5 = {.shape = {1, 1, 1, 128} , .stride = out_cast_3.stride, .addr = out_cast_3.addr + ((0) * out_cast_3.stride.c+(0) * out_cast_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_3.stride.c+(0) * out_cast_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_3.default_stride};
    __ppl_tensor_info OUT_1 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(128) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(128) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_1.addr, out_cast_5.addr, &OUT_1.shape, (OUT_1.default_stride ? NULL : &OUT_1.stride), (out_cast_5.default_stride ? NULL : &out_cast_5.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_3.stride, 0, &q_upper_acc_h0_3.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_4.stride, 0, &q_upper_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_3.stride, 0, &q_upper_acc_h1_3.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_4.stride, 0, &q_upper_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_2.stride, 0, &q_rope_src_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_2.stride, 0, &q_rope_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_1.stride, 0, &q_rope_tmp0_h0_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_1.stride, 0, &q_rope_tmp1_h0_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_1.stride, 0, &q_rope_tmp2_h0_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_1.stride, 0, &q_rope_tmp3_h0_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_2.stride, 0, &q_rope_src_h1_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_2.stride, 0, &q_rope_h1_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_1.stride, 0, &q_rope_tmp0_h1_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_1.stride, 0, &q_rope_tmp1_h1_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_1.stride, 0, &q_rope_tmp2_h1_1.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_1.stride, 0, &q_rope_tmp3_h1_1.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_2.stride, 0, &cos_shared_2.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_2.stride, 0, &sin_shared_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_2.stride, 0, &pe_cur_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_1.stride, 0, &pe_cur_cos_1.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_1.stride, 0, &pe_cur_sin_1.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_1.stride, 0, &pe_cur_neg_1.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_1 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_1.stride, 0, &pe_cur_neg_sin_1.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_2.stride, 0, &pe_cur_rope_2.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_6 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_6.stride, 0, &wukv_full_tile_6.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_4 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_4.stride, 0, &kv_cache_full_4.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_4 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_4.stride, 0, &pe_score_tile_4.shape, DT_BFP16);
  __ppl_tensor_info mask_row_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_2.stride, 0, &mask_row_2.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_3.stride, 0, &score_latent_h0_3.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_3.stride, 0, &score_rope_h0_3.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_1.stride, 0, &score_latent_h1_1.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_1 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_1.stride, 0, &score_rope_h1_1.shape, DT_FP32);
  if (tpu_workitem_index() == 1) {
    __ppl_tensor_info COS_1 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_3 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_2.stride, .addr = cos_shared_2.addr + ((0) * cos_shared_2.stride.c+(0) * cos_shared_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_2.stride.c+(0) * cos_shared_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_2.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_3.addr, COS_1.addr, &cos_shared_3.shape, (cos_shared_3.default_stride ? NULL : &cos_shared_3.stride), (COS_1.default_stride ? NULL : &COS_1.stride), DT_BFP16);
    __ppl_tensor_info SIN_1 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_3 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_2.stride, .addr = sin_shared_2.addr + ((0) * sin_shared_2.stride.c+(0) * sin_shared_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_2.stride.c+(0) * sin_shared_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_2.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_3.addr, SIN_1.addr, &sin_shared_3.shape, (sin_shared_3.default_stride ? NULL : &sin_shared_3.stride), (SIN_1.default_stride ? NULL : &SIN_1.stride), DT_BFP16);
    for (int d_2 = 0; d_2 < 4; ++d_2) {
      __ppl_tensor_info kv_cur_tile_3 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_3.stride, 0, &kv_cur_tile_3.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_1 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_2 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_2 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_4 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_3.stride, .addr = kv_cur_tile_3.addr + ((0) * kv_cur_tile_3.stride.c+(0) * kv_cur_tile_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_3.stride.c+(0) * kv_cur_tile_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_3.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_4.addr, KV_CUR_1.addr, &kv_cur_tile_4.shape, (kv_cur_tile_4.default_stride ? NULL : &kv_cur_tile_4.stride), (KV_CUR_1.default_stride ? NULL : &KV_CUR_1.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_5 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_3.stride, .addr = kv_cur_tile_3.addr + ((0) * kv_cur_tile_3.stride.c+(0) * kv_cur_tile_3.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_3.stride.c+(0) * kv_cur_tile_3.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_3.default_stride};
      __ppl_tensor_info KVcache_OUT_2 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_2 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_2 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_2.addr, kv_cur_tile_5.addr, &KVcache_OUT_2.shape, (KVcache_OUT_2.default_stride ? NULL : &KVcache_OUT_2.stride), (kv_cur_tile_5.default_stride ? NULL : &kv_cur_tile_5.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_1 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_3 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_2.stride, .addr = pe_cur_2.addr + ((0) * pe_cur_2.stride.c+(0) * pe_cur_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_2.stride.c+(0) * pe_cur_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_2.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_3.addr, PE_CUR_1.addr, &pe_cur_3.shape, (pe_cur_3.default_stride ? NULL : &pe_cur_3.stride), (PE_CUR_1.default_stride ? NULL : &PE_CUR_1.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_1.addr, pe_cur_2.addr, cos_shared_2.addr, &pe_cur_cos_1.shape, (pe_cur_cos_1.default_stride ? NULL : &pe_cur_cos_1.stride), (pe_cur_2.default_stride ? NULL : &pe_cur_2.stride), (cos_shared_2.default_stride ? NULL : &cos_shared_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_1.addr, pe_cur_2.addr, sin_shared_2.addr, &pe_cur_sin_1.shape, (pe_cur_sin_1.default_stride ? NULL : &pe_cur_sin_1.stride), (pe_cur_2.default_stride ? NULL : &pe_cur_2.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_1_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_1_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_1_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_1.addr, pe_cur_2.addr, pe_cur_neg_1_scalar_DT_BFP16, &pe_cur_neg_1.shape, (pe_cur_neg_1.default_stride ? NULL : &pe_cur_neg_1.stride), (pe_cur_2.default_stride ? NULL : &pe_cur_2.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_1.addr, pe_cur_neg_1.addr, sin_shared_2.addr, &pe_cur_neg_sin_1.shape, (pe_cur_neg_sin_1.default_stride ? NULL : &pe_cur_neg_sin_1.stride), (pe_cur_neg_1.default_stride ? NULL : &pe_cur_neg_1.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_2.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_2.shape.n, .c = pe_cur_rope_2.shape.c, .h = pe_cur_rope_2.shape.h, .w = pe_cur_rope_2.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_2.addr, pe_cur_cos_1.addr, pe_cur_neg_sin_1.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_2.addr + 2, pe_cur_cos_1.addr + 2, pe_cur_sin_1.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_3 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_2.stride, .addr = pe_cur_rope_2.addr + ((0) * pe_cur_rope_2.stride.c+(0) * pe_cur_rope_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_2.stride.c+(0) * pe_cur_rope_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_2.default_stride};
    __ppl_tensor_info PEcache_OUT_2 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_2.addr, pe_cur_rope_3.addr, &PEcache_OUT_2.shape, (PEcache_OUT_2.default_stride ? NULL : &PEcache_OUT_2.stride), (pe_cur_rope_3.default_stride ? NULL : &pe_cur_rope_3.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_3.addr, q_upper_acc_h0_3_scalar_f32, &q_upper_acc_h0_3.shape, (q_upper_acc_h0_3.default_stride ? NULL : &q_upper_acc_h0_3.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_3.addr, q_upper_acc_h1_3_scalar_f32, &q_upper_acc_h1_3.shape, (q_upper_acc_h1_3.default_stride ? NULL : &q_upper_acc_h1_3.stride), DT_FP32);
    }
    for (int k_1 = 0; k_1 < 12; ++k_1) {
      __ppl_tensor_info q_tile_4 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_4.stride, 0, &q_tile_4.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_3 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_3.stride, 0, &wuq_fp8_3.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_8 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_8.stride, 0, &wuq_scale_stage_8.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_4 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_4.stride, 0, &wuq_tile_4.shape, DT_BFP16);
      __ppl_tensor_info Q_1 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_1 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_1 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_5 = {.shape = {1, 1, 1, 128} , .stride = q_tile_4.stride, .addr = q_tile_4.addr + ((0) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_4.default_stride};
      tpu_gdma_cpy_S2L(q_tile_5.addr, Q_1.addr, &q_tile_5.shape, (q_tile_5.default_stride ? NULL : &q_tile_5.stride), (Q_1.default_stride ? NULL : &Q_1.stride), DT_BFP16);
      __ppl_tensor_info WUQ_1 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((384) * 1536+((k_1 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((384) * 1536+((k_1 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_4 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_3.stride, .addr = wuq_fp8_3.addr + ((0) * wuq_fp8_3.stride.c+(0) * wuq_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_3.stride.c+(0) * wuq_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_3.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_4.addr, WUQ_1.addr, &wuq_fp8_4.shape, (wuq_fp8_4.default_stride ? NULL : &wuq_fp8_4.stride), (WUQ_1.default_stride ? NULL : &WUQ_1.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_18 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_18.stride, 0, &wuq_scale_scalar_18.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_6 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((3) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((3) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_19 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_18.stride, .addr = wuq_scale_scalar_18.addr + ((0) * wuq_scale_scalar_18.stride.c+(0) * wuq_scale_scalar_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_18.stride.c+(0) * wuq_scale_scalar_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_18.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_19.addr, WUQ_SCALE_6.addr, &wuq_scale_scalar_19.shape, (wuq_scale_scalar_19.default_stride ? NULL : &wuq_scale_scalar_19.stride), (WUQ_SCALE_6.default_stride ? NULL : &WUQ_SCALE_6.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_20 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_18.stride, .addr = wuq_scale_scalar_18.addr + (((0) / 64) * wuq_scale_scalar_18.stride.c+(0) * wuq_scale_scalar_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_18.stride.c+(0) * wuq_scale_scalar_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_9 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((0) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_9.addr, wuq_scale_scalar_20.addr, &wuq_scale_stage_9.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_21 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_21.stride, 0, &wuq_scale_scalar_21.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_7 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((3) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((3) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_22 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_21.stride, .addr = wuq_scale_scalar_21.addr + ((0) * wuq_scale_scalar_21.stride.c+(0) * wuq_scale_scalar_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_21.stride.c+(0) * wuq_scale_scalar_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_21.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_22.addr, WUQ_SCALE_7.addr, &wuq_scale_scalar_22.shape, (wuq_scale_scalar_22.default_stride ? NULL : &wuq_scale_scalar_22.stride), (WUQ_SCALE_7.default_stride ? NULL : &WUQ_SCALE_7.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_23 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_21.stride, .addr = wuq_scale_scalar_21.addr + (((0) / 64) * wuq_scale_scalar_21.stride.c+(0) * wuq_scale_scalar_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_21.stride.c+(0) * wuq_scale_scalar_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_10 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((64) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_10.addr, wuq_scale_scalar_23.addr, &wuq_scale_stage_10.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_24 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_24.stride, 0, &wuq_scale_scalar_24.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_8 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((4) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((4) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_25 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_24.stride, .addr = wuq_scale_scalar_24.addr + ((0) * wuq_scale_scalar_24.stride.c+(0) * wuq_scale_scalar_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_24.stride.c+(0) * wuq_scale_scalar_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_24.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_25.addr, WUQ_SCALE_8.addr, &wuq_scale_scalar_25.shape, (wuq_scale_scalar_25.default_stride ? NULL : &wuq_scale_scalar_25.stride), (WUQ_SCALE_8.default_stride ? NULL : &WUQ_SCALE_8.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_26 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_24.stride, .addr = wuq_scale_scalar_24.addr + (((0) / 64) * wuq_scale_scalar_24.stride.c+(0) * wuq_scale_scalar_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_24.stride.c+(0) * wuq_scale_scalar_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_11 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((128) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_11.addr, wuq_scale_scalar_26.addr, &wuq_scale_stage_11.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_27 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_27.stride, 0, &wuq_scale_scalar_27.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_9 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((4) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((4) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_28 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_27.stride, .addr = wuq_scale_scalar_27.addr + ((0) * wuq_scale_scalar_27.stride.c+(0) * wuq_scale_scalar_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_27.stride.c+(0) * wuq_scale_scalar_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_27.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_28.addr, WUQ_SCALE_9.addr, &wuq_scale_scalar_28.shape, (wuq_scale_scalar_28.default_stride ? NULL : &wuq_scale_scalar_28.stride), (WUQ_SCALE_9.default_stride ? NULL : &WUQ_SCALE_9.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_29 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_27.stride, .addr = wuq_scale_scalar_27.addr + (((0) / 64) * wuq_scale_scalar_27.stride.c+(0) * wuq_scale_scalar_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_27.stride.c+(0) * wuq_scale_scalar_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_12 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((192) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_12.addr, wuq_scale_scalar_29.addr, &wuq_scale_stage_12.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_30 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_30.stride, 0, &wuq_scale_scalar_30.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_10 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((5) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((5) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_31 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_30.stride, .addr = wuq_scale_scalar_30.addr + ((0) * wuq_scale_scalar_30.stride.c+(0) * wuq_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_30.stride.c+(0) * wuq_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_30.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_31.addr, WUQ_SCALE_10.addr, &wuq_scale_scalar_31.shape, (wuq_scale_scalar_31.default_stride ? NULL : &wuq_scale_scalar_31.stride), (WUQ_SCALE_10.default_stride ? NULL : &WUQ_SCALE_10.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_32 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_30.stride, .addr = wuq_scale_scalar_30.addr + (((0) / 64) * wuq_scale_scalar_30.stride.c+(0) * wuq_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_30.stride.c+(0) * wuq_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_13 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((256) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_13.addr, wuq_scale_scalar_32.addr, &wuq_scale_stage_13.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_33 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_33.stride, 0, &wuq_scale_scalar_33.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_11 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((5) * 12+(k_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((5) * 12+(k_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_34 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_33.stride, .addr = wuq_scale_scalar_33.addr + ((0) * wuq_scale_scalar_33.stride.c+(0) * wuq_scale_scalar_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_33.stride.c+(0) * wuq_scale_scalar_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_33.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_34.addr, WUQ_SCALE_11.addr, &wuq_scale_scalar_34.shape, (wuq_scale_scalar_34.default_stride ? NULL : &wuq_scale_scalar_34.stride), (WUQ_SCALE_11.default_stride ? NULL : &WUQ_SCALE_11.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_35 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_33.stride, .addr = wuq_scale_scalar_33.addr + (((0) / 64) * wuq_scale_scalar_33.stride.c+(0) * wuq_scale_scalar_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_33.stride.c+(0) * wuq_scale_scalar_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_14 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((320) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_14.addr, wuq_scale_scalar_35.addr, &wuq_scale_stage_14.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_5 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_3.stride, .addr = wuq_fp8_3.addr + (((0) / 64) * wuq_fp8_3.stride.c+(0) * wuq_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_3.stride.c+(0) * wuq_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_15 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_8.stride, .addr = wuq_scale_stage_8.addr + (((0) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_8.stride.c+(0) * wuq_scale_stage_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_5 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_4.stride, .addr = wuq_tile_4.addr + (((0) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_5.addr, wuq_fp8_5.addr, &wuq_tile_5.shape, (wuq_tile_5.default_stride ? NULL : &wuq_tile_5.stride), (wuq_fp8_5.default_stride ? NULL : &wuq_fp8_5.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_15_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_15_block_stride, 0, &wuq_scale_stage_15.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_15_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_5.addr, wuq_tile_5.addr, wuq_scale_stage_15.addr, &wuq_tile_5.shape, (wuq_tile_5.default_stride ? NULL : &wuq_tile_5.stride), (wuq_tile_5.default_stride ? NULL : &wuq_tile_5.stride), &wuq_scale_stage_15_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_6 = {.shape = {1, 1, 1, 128}, .stride = q_tile_4.stride, .addr = q_tile_4.addr + (((0) / 64) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_6 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_4.stride, .addr = wuq_tile_4.addr + (((0) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_4 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_3.stride, .addr = q_upper_acc_h0_3.addr + (((0) / 64) * q_upper_acc_h0_3.stride.c+(0) * q_upper_acc_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_3.stride.c+(0) * q_upper_acc_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_4.addr, q_tile_6.addr, wuq_tile_6.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_7 = {.shape = {1, 1, 1, 128}, .stride = q_tile_4.stride, .addr = q_tile_4.addr + (((0) / 64) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_4.stride.c+(0) * q_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_7 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_4.stride, .addr = wuq_tile_4.addr + (((192) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_4.stride.c+(0) * wuq_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_4 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_3.stride, .addr = q_upper_acc_h1_3.addr + (((0) / 64) * q_upper_acc_h1_3.stride.c+(0) * q_upper_acc_h1_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_3.stride.c+(0) * q_upper_acc_h1_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_4.addr, q_tile_7.addr, wuq_tile_7.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_5 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_3.stride, .addr = q_upper_acc_h0_3.addr + ((0) * q_upper_acc_h0_3.stride.c+(0) * q_upper_acc_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_3.stride.c+(0) * q_upper_acc_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_3.default_stride};
    __ppl_tensor_info q_upper_h0_5 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_4.stride, .addr = q_upper_h0_4.addr + ((0) * q_upper_h0_4.stride.c+(0) * q_upper_h0_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_4.stride.c+(0) * q_upper_h0_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_4.default_stride};
    tpu_bdc_cast(q_upper_h0_5.addr, q_upper_acc_h0_5.addr, &q_upper_h0_5.shape, (q_upper_h0_5.default_stride ? NULL : &q_upper_h0_5.stride), (q_upper_acc_h0_5.default_stride ? NULL : &q_upper_acc_h0_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_5 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_3.stride, .addr = q_upper_acc_h1_3.addr + ((0) * q_upper_acc_h1_3.stride.c+(0) * q_upper_acc_h1_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_3.stride.c+(0) * q_upper_acc_h1_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_3.default_stride};
    __ppl_tensor_info q_upper_h1_5 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_4.stride, .addr = q_upper_h1_4.addr + ((0) * q_upper_h1_4.stride.c+(0) * q_upper_h1_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_4.stride.c+(0) * q_upper_h1_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_4.default_stride};
    tpu_bdc_cast(q_upper_h1_5.addr, q_upper_acc_h1_5.addr, &q_upper_h1_5.shape, (q_upper_h1_5.default_stride ? NULL : &q_upper_h1_5.stride), (q_upper_acc_h1_5.default_stride ? NULL : &q_upper_acc_h1_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_3 = 0; d_3 < 4; ++d_3) {
      __ppl_tensor_info wukv_fp8_3 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_3.stride, 0, &wukv_fp8_3.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_6 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_6.stride, 0, &wukv_scale_stage_6.shape, DT_BFP16);
      __ppl_tensor_info WUKV_1 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((512) * 512+((d_3 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((512) * 512+((d_3 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_4 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_3.stride, .addr = wukv_fp8_3.addr + ((0) * wukv_fp8_3.stride.c+(0) * wukv_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_3.stride.c+(0) * wukv_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_3.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_4.addr, WUKV_1.addr, &wukv_fp8_4.shape, (wukv_fp8_4.default_stride ? NULL : &wukv_fp8_4.stride), (WUKV_1.default_stride ? NULL : &WUKV_1.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_10 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_10.stride, 0, &wukv_scale_scalar_10.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_4 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((4) * 4+(d_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((4) * 4+(d_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_11 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_10.stride, .addr = wukv_scale_scalar_10.addr + ((0) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_10.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_11.addr, WUKV_SCALE_4.addr, &wukv_scale_scalar_11.shape, (wukv_scale_scalar_11.default_stride ? NULL : &wukv_scale_scalar_11.stride), (WUKV_SCALE_4.default_stride ? NULL : &WUKV_SCALE_4.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_12 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_10.stride, .addr = wukv_scale_scalar_10.addr + (((0) / 64) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_7 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_6.stride, .addr = wukv_scale_stage_6.addr + (((0) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_7_chunk = wukv_scale_stage_7;
      wukv_scale_stage_7_chunk.shape.c = 64;
      wukv_scale_stage_7_chunk.addr = wukv_scale_stage_7.addr + ((0) * wukv_scale_stage_7.stride.c) * 2;
      wukv_scale_stage_7_chunk.offset = wukv_scale_stage_7.offset + ((0) * wukv_scale_stage_7.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_7_chunk.addr, wukv_scale_scalar_12.addr, &wukv_scale_stage_7_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_7_chunk_1 = wukv_scale_stage_7;
      wukv_scale_stage_7_chunk_1.shape.c = 64;
      wukv_scale_stage_7_chunk_1.addr = wukv_scale_stage_7.addr + ((1) * wukv_scale_stage_7.stride.c) * 2;
      wukv_scale_stage_7_chunk_1.offset = wukv_scale_stage_7.offset + ((1) * wukv_scale_stage_7.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_7_chunk_1.addr, wukv_scale_scalar_12.addr, &wukv_scale_stage_7_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_5 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((5) * 4+(d_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((5) * 4+(d_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_13 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_10.stride, .addr = wukv_scale_scalar_10.addr + ((0) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_10.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_13.addr, WUKV_SCALE_5.addr, &wukv_scale_scalar_13.shape, (wukv_scale_scalar_13.default_stride ? NULL : &wukv_scale_scalar_13.stride), (WUKV_SCALE_5.default_stride ? NULL : &WUKV_SCALE_5.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_14 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_10.stride, .addr = wukv_scale_scalar_10.addr + (((0) / 64) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_10.stride.c+(0) * wukv_scale_scalar_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_8 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_6.stride, .addr = wukv_scale_stage_6.addr + (((128) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_8_chunk = wukv_scale_stage_8;
      wukv_scale_stage_8_chunk.shape.c = 64;
      wukv_scale_stage_8_chunk.addr = wukv_scale_stage_8.addr + ((0) * wukv_scale_stage_8.stride.c) * 2;
      wukv_scale_stage_8_chunk.offset = wukv_scale_stage_8.offset + ((0) * wukv_scale_stage_8.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_8_chunk.addr, wukv_scale_scalar_14.addr, &wukv_scale_stage_8_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_8_chunk_1 = wukv_scale_stage_8;
      wukv_scale_stage_8_chunk_1.shape.c = 64;
      wukv_scale_stage_8_chunk_1.addr = wukv_scale_stage_8.addr + ((1) * wukv_scale_stage_8.stride.c) * 2;
      wukv_scale_stage_8_chunk_1.offset = wukv_scale_stage_8.offset + ((1) * wukv_scale_stage_8.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_8_chunk_1.addr, wukv_scale_scalar_14.addr, &wukv_scale_stage_8_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_15 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_15.stride, 0, &wukv_scale_scalar_15.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_6 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((6) * 4+(d_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((6) * 4+(d_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_16 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_15.stride, .addr = wukv_scale_scalar_15.addr + ((0) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_15.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_16.addr, WUKV_SCALE_6.addr, &wukv_scale_scalar_16.shape, (wukv_scale_scalar_16.default_stride ? NULL : &wukv_scale_scalar_16.stride), (WUKV_SCALE_6.default_stride ? NULL : &WUKV_SCALE_6.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_17 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_15.stride, .addr = wukv_scale_scalar_15.addr + (((0) / 64) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_9 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_6.stride, .addr = wukv_scale_stage_6.addr + (((256) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_9_chunk = wukv_scale_stage_9;
      wukv_scale_stage_9_chunk.shape.c = 64;
      wukv_scale_stage_9_chunk.addr = wukv_scale_stage_9.addr + ((0) * wukv_scale_stage_9.stride.c) * 2;
      wukv_scale_stage_9_chunk.offset = wukv_scale_stage_9.offset + ((0) * wukv_scale_stage_9.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_9_chunk.addr, wukv_scale_scalar_17.addr, &wukv_scale_stage_9_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_9_chunk_1 = wukv_scale_stage_9;
      wukv_scale_stage_9_chunk_1.shape.c = 64;
      wukv_scale_stage_9_chunk_1.addr = wukv_scale_stage_9.addr + ((1) * wukv_scale_stage_9.stride.c) * 2;
      wukv_scale_stage_9_chunk_1.offset = wukv_scale_stage_9.offset + ((1) * wukv_scale_stage_9.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_9_chunk_1.addr, wukv_scale_scalar_17.addr, &wukv_scale_stage_9_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_7 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((7) * 4+(d_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((7) * 4+(d_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_18 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_15.stride, .addr = wukv_scale_scalar_15.addr + ((0) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_15.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_18.addr, WUKV_SCALE_7.addr, &wukv_scale_scalar_18.shape, (wukv_scale_scalar_18.default_stride ? NULL : &wukv_scale_scalar_18.stride), (WUKV_SCALE_7.default_stride ? NULL : &WUKV_SCALE_7.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_19 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_15.stride, .addr = wukv_scale_scalar_15.addr + (((0) / 64) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_15.stride.c+(0) * wukv_scale_scalar_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_10 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_6.stride, .addr = wukv_scale_stage_6.addr + (((384) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_10_chunk = wukv_scale_stage_10;
      wukv_scale_stage_10_chunk.shape.c = 64;
      wukv_scale_stage_10_chunk.addr = wukv_scale_stage_10.addr + ((0) * wukv_scale_stage_10.stride.c) * 2;
      wukv_scale_stage_10_chunk.offset = wukv_scale_stage_10.offset + ((0) * wukv_scale_stage_10.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_10_chunk.addr, wukv_scale_scalar_19.addr, &wukv_scale_stage_10_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_10_chunk_1 = wukv_scale_stage_10;
      wukv_scale_stage_10_chunk_1.shape.c = 64;
      wukv_scale_stage_10_chunk_1.addr = wukv_scale_stage_10.addr + ((1) * wukv_scale_stage_10.stride.c) * 2;
      wukv_scale_stage_10_chunk_1.offset = wukv_scale_stage_10.offset + ((1) * wukv_scale_stage_10.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_10_chunk_1.addr, wukv_scale_scalar_19.addr, &wukv_scale_stage_10_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_5 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_3.stride, .addr = wukv_fp8_3.addr + (((0) / 64) * wukv_fp8_3.stride.c+(0) * wukv_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_3.stride.c+(0) * wukv_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_11 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_6.stride, .addr = wukv_scale_stage_6.addr + (((0) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_6.stride.c+(0) * wukv_scale_stage_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_7 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_6.stride, .addr = wukv_full_tile_6.addr + (((0) / 64) * wukv_full_tile_6.stride.c+((d_3 * 128)) * wukv_full_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_6.stride.c+((d_3 * 128)) * wukv_full_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_7.addr, wukv_fp8_5.addr, &wukv_full_tile_7.shape, (wukv_full_tile_7.default_stride ? NULL : &wukv_full_tile_7.stride), (wukv_fp8_5.default_stride ? NULL : &wukv_fp8_5.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_11_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_11_block_stride, 0, &wukv_scale_stage_11.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_11_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_7.addr, wukv_full_tile_7.addr, wukv_scale_stage_11.addr, &wukv_full_tile_7.shape, (wukv_full_tile_7.default_stride ? NULL : &wukv_full_tile_7.stride), (wukv_full_tile_7.default_stride ? NULL : &wukv_full_tile_7.stride), &wukv_scale_stage_11_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_3 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_5 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_4.stride, .addr = kv_cache_full_4.addr + ((0) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_4.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_5.addr, KVcache_OUT_3.addr, &kv_cache_full_5.shape, (kv_cache_full_5.default_stride ? NULL : &kv_cache_full_5.stride), (KVcache_OUT_3.default_stride ? NULL : &KVcache_OUT_3.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_3 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_5 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_4.stride, .addr = pe_score_tile_4.addr + ((0) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_4.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_5.addr, PEcache_OUT_3.addr, &pe_score_tile_5.shape, (pe_score_tile_5.default_stride ? NULL : &pe_score_tile_5.stride), (PEcache_OUT_3.default_stride ? NULL : &PEcache_OUT_3.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_1 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_3 = {.shape = {1, 1, 1, 192} , .stride = mask_row_2.stride, .addr = mask_row_2.addr + ((0) * mask_row_2.stride.c+(0) * mask_row_2.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_2.stride.c+(0) * mask_row_2.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_2.default_stride};
    tpu_gdma_cpy_S2L(mask_row_3.addr, DECODE_MASK_1.addr, &mask_row_3.shape, (mask_row_3.default_stride ? NULL : &mask_row_3.stride), (DECODE_MASK_1.default_stride ? NULL : &DECODE_MASK_1.stride), DT_FP32);
    {
    scalar_t score_latent_h0_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_3.addr, score_latent_h0_3_scalar_f32, &score_latent_h0_3.shape, (score_latent_h0_3.default_stride ? NULL : &score_latent_h0_3.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_3.addr, score_rope_h0_3_scalar_f32, &score_rope_h0_3.shape, (score_rope_h0_3.default_stride ? NULL : &score_rope_h0_3.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_1.addr, score_latent_h1_1_scalar_f32, &score_latent_h1_1.shape, (score_latent_h1_1.default_stride ? NULL : &score_latent_h1_1.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_1.addr, score_rope_h1_1_scalar_f32, &score_rope_h1_1.shape, (score_rope_h1_1.default_stride ? NULL : &score_rope_h1_1.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_6 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_4.stride, .addr = q_upper_h0_4.addr + ((0) * q_upper_h0_4.stride.c+(128) * q_upper_h0_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_4.stride.c+(128) * q_upper_h0_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_4.default_stride};
    __ppl_tensor_info q_rope_src_h0_3 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_2.stride, .addr = q_rope_src_h0_2.addr + ((0) * q_rope_src_h0_2.stride.c+(0) * q_rope_src_h0_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_2.stride.c+(0) * q_rope_src_h0_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_2.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_3.addr, q_upper_h0_6.addr, &q_rope_src_h0_3.shape, (q_rope_src_h0_3.default_stride ? NULL : &q_rope_src_h0_3.stride), (q_upper_h0_6.default_stride ? NULL : &q_upper_h0_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_1.addr, q_rope_src_h0_2.addr, cos_shared_2.addr, &q_rope_tmp0_h0_1.shape, (q_rope_tmp0_h0_1.default_stride ? NULL : &q_rope_tmp0_h0_1.stride), (q_rope_src_h0_2.default_stride ? NULL : &q_rope_src_h0_2.stride), (cos_shared_2.default_stride ? NULL : &cos_shared_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_1.addr, q_rope_src_h0_2.addr, sin_shared_2.addr, &q_rope_tmp1_h0_1.shape, (q_rope_tmp1_h0_1.default_stride ? NULL : &q_rope_tmp1_h0_1.stride), (q_rope_src_h0_2.default_stride ? NULL : &q_rope_src_h0_2.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_1_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_1_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_1_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_1.addr, q_rope_src_h0_2.addr, q_rope_tmp2_h0_1_scalar_DT_BFP16, &q_rope_tmp2_h0_1.shape, (q_rope_tmp2_h0_1.default_stride ? NULL : &q_rope_tmp2_h0_1.stride), (q_rope_src_h0_2.default_stride ? NULL : &q_rope_src_h0_2.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_1.addr, q_rope_tmp2_h0_1.addr, sin_shared_2.addr, &q_rope_tmp3_h0_1.shape, (q_rope_tmp3_h0_1.default_stride ? NULL : &q_rope_tmp3_h0_1.stride), (q_rope_tmp2_h0_1.default_stride ? NULL : &q_rope_tmp2_h0_1.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_2.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_2.shape.n, .c = q_rope_h0_2.shape.c, .h = q_rope_h0_2.shape.h, .w = q_rope_h0_2.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_2.addr, q_rope_tmp0_h0_1.addr, q_rope_tmp3_h0_1.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_2.addr + 2, q_rope_tmp0_h0_1.addr + 2, q_rope_tmp1_h0_1.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_6 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_4.stride, .addr = q_upper_h1_4.addr + ((0) * q_upper_h1_4.stride.c+(128) * q_upper_h1_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_4.stride.c+(128) * q_upper_h1_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_4.default_stride};
    __ppl_tensor_info q_rope_src_h1_3 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_2.stride, .addr = q_rope_src_h1_2.addr + ((0) * q_rope_src_h1_2.stride.c+(0) * q_rope_src_h1_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_2.stride.c+(0) * q_rope_src_h1_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_2.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_3.addr, q_upper_h1_6.addr, &q_rope_src_h1_3.shape, (q_rope_src_h1_3.default_stride ? NULL : &q_rope_src_h1_3.stride), (q_upper_h1_6.default_stride ? NULL : &q_upper_h1_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_1.addr, q_rope_src_h1_2.addr, cos_shared_2.addr, &q_rope_tmp0_h1_1.shape, (q_rope_tmp0_h1_1.default_stride ? NULL : &q_rope_tmp0_h1_1.stride), (q_rope_src_h1_2.default_stride ? NULL : &q_rope_src_h1_2.stride), (cos_shared_2.default_stride ? NULL : &cos_shared_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_1.addr, q_rope_src_h1_2.addr, sin_shared_2.addr, &q_rope_tmp1_h1_1.shape, (q_rope_tmp1_h1_1.default_stride ? NULL : &q_rope_tmp1_h1_1.stride), (q_rope_src_h1_2.default_stride ? NULL : &q_rope_src_h1_2.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_1_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_1_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_1_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_1.addr, q_rope_src_h1_2.addr, q_rope_tmp2_h1_1_scalar_DT_BFP16, &q_rope_tmp2_h1_1.shape, (q_rope_tmp2_h1_1.default_stride ? NULL : &q_rope_tmp2_h1_1.stride), (q_rope_src_h1_2.default_stride ? NULL : &q_rope_src_h1_2.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_1.addr, q_rope_tmp2_h1_1.addr, sin_shared_2.addr, &q_rope_tmp3_h1_1.shape, (q_rope_tmp3_h1_1.default_stride ? NULL : &q_rope_tmp3_h1_1.stride), (q_rope_tmp2_h1_1.default_stride ? NULL : &q_rope_tmp2_h1_1.stride), (sin_shared_2.default_stride ? NULL : &sin_shared_2.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_2.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_2.shape.n, .c = q_rope_h1_2.shape.c, .h = q_rope_h1_2.shape.h, .w = q_rope_h1_2.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_2.addr, q_rope_tmp0_h1_1.addr, q_rope_tmp3_h1_1.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_2.addr + 2, q_rope_tmp0_h1_1.addr + 2, q_rope_tmp1_h1_1.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_6 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_6.stride, 0, &q_nope_head_6.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_6 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_6.stride, 0, &q_wukv_acc_6.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_6 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_6.stride, 0, &q_wukv_head_6.shape, DT_BFP16);
    __ppl_tensor_info score_tile_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_4.stride, 0, &score_tile_4.shape, DT_FP32);
    __ppl_tensor_info prob_tile_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_4.stride, 0, &prob_tile_4.shape, DT_BFP16);
    __ppl_tensor_info scores_max_2 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_2.stride, 0, &scores_max_2.shape, DT_FP32);
    __ppl_tensor_info scores_sum_2 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_2.stride, 0, &scores_sum_2.shape, DT_FP32);
    __ppl_tensor_info work_score0_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_2.stride, 0, &work_score0_2.shape, DT_FP32);
    __ppl_tensor_info work_score1_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_2.stride, 0, &work_score1_2.shape, DT_FP32);
    __ppl_tensor_info coeff_2 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_2.stride, 0, &coeff_2.shape, DT_FP32);
    __ppl_tensor_info table_2 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_2.stride, 0, &table_2.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_4 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_4.stride, 0, &ctx_acc_4.shape, DT_FP32);
    __ppl_tensor_info ctx_head_6 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_6.stride, 0, &ctx_head_6.shape, DT_BFP16);
    __ppl_tensor_info out_part_6 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_6.stride, 0, &out_part_6.shape, DT_FP32);
    __ppl_tensor_info out_cast_6 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_6.stride, 0, &out_cast_6.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_7 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_4.stride, .addr = q_upper_h0_4.addr + ((0) * q_upper_h0_4.stride.c+(0) * q_upper_h0_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_4.stride.c+(0) * q_upper_h0_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_4.default_stride};
    __ppl_tensor_info q_nope_head_7 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_6.stride, .addr = q_nope_head_6.addr + ((0) * q_nope_head_6.stride.c+(0) * q_nope_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_6.stride.c+(0) * q_nope_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_6.default_stride};
    tpu_bdc_cpy(q_nope_head_7.addr, q_upper_h0_7.addr, &q_nope_head_7.shape, (q_nope_head_7.default_stride ? NULL : &q_nope_head_7.stride), (q_upper_h0_7.default_stride ? NULL : &q_upper_h0_7.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_6.addr, q_wukv_acc_6_scalar_f32, &q_wukv_acc_6.shape, (q_wukv_acc_6.default_stride ? NULL : &q_wukv_acc_6.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_8 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_6.stride, .addr = q_nope_head_6.addr + (((0) / 64) * q_nope_head_6.stride.c+(0) * q_nope_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_6.stride.c+(0) * q_nope_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_8 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_6.stride, .addr = wukv_full_tile_6.addr + (((0) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_7 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_6.stride, .addr = q_wukv_acc_6.addr + (((0) / 64) * q_wukv_acc_6.stride.c+(0) * q_wukv_acc_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_6.stride.c+(0) * q_wukv_acc_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_7.addr, q_nope_head_8.addr, wukv_full_tile_8.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_8 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_6.stride, .addr = q_wukv_acc_6.addr + ((0) * q_wukv_acc_6.stride.c+(0) * q_wukv_acc_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_6.stride.c+(0) * q_wukv_acc_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_6.default_stride};
    __ppl_tensor_info q_wukv_head_7 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_6.stride, .addr = q_wukv_head_6.addr + ((0) * q_wukv_head_6.stride.c+(0) * q_wukv_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_6.stride.c+(0) * q_wukv_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_6.default_stride};
    tpu_bdc_cast(q_wukv_head_7.addr, q_wukv_acc_8.addr, &q_wukv_head_7.shape, (q_wukv_head_7.default_stride ? NULL : &q_wukv_head_7.stride), (q_wukv_acc_8.default_stride ? NULL : &q_wukv_acc_8.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_8 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_6.stride, .addr = q_wukv_head_6.addr + (((0) / 64) * q_wukv_head_6.stride.c+(0) * q_wukv_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_6.stride.c+(0) * q_wukv_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_6 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_4.stride, .addr = kv_cache_full_4.addr + (((0) / 64) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_4 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_3.stride, .addr = score_latent_h0_3.addr + (((0) / 64) * score_latent_h0_3.stride.c+(0) * score_latent_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_3.stride.c+(0) * score_latent_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_4.addr, q_wukv_head_8.addr, kv_cache_full_6.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_3 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_2.stride, .addr = q_rope_h0_2.addr + (((0) / 64) * q_rope_h0_2.stride.c+(0) * q_rope_h0_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_2.stride.c+(0) * q_rope_h0_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_6 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_4.stride, .addr = pe_score_tile_4.addr + (((0) / 64) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_4 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_3.stride, .addr = score_rope_h0_3.addr + (((0) / 64) * score_rope_h0_3.stride.c+(0) * score_rope_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_3.stride.c+(0) * score_rope_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_4.addr, q_rope_h0_3.addr, pe_score_tile_6.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_4.addr, score_latent_h0_3.addr, score_rope_h0_3.addr, &score_tile_4.shape, (score_tile_4.default_stride ? NULL : &score_tile_4.stride), (score_latent_h0_3.default_stride ? NULL : &score_latent_h0_3.stride), (score_rope_h0_3.default_stride ? NULL : &score_rope_h0_3.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_4.addr, score_tile_4.addr, mask_row_2.addr, &score_tile_4.shape, (score_tile_4.default_stride ? NULL : &score_tile_4.stride), (score_tile_4.default_stride ? NULL : &score_tile_4.stride), (mask_row_2.default_stride ? NULL : &mask_row_2.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_4.addr, score_tile_4.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_4.shape, (score_tile_4.default_stride ? NULL : &score_tile_4.stride), (score_tile_4.default_stride ? NULL : &score_tile_4.stride), DT_FP32);
    {
    scalar_t scores_max_2_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_2.addr, scores_max_2_scalar_f32, &scores_max_2.shape, (scores_max_2.default_stride ? NULL : &scores_max_2.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_2 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_2.stride, 0, &tmp_buffer_max_2.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_4.shape.n, score_tile_4.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_4.shape.n, score_tile_4.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_4.shape.n, score_tile_4.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_4.shape.n, score_tile_4.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_4.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, score_tile_4.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_4.shape.w) {
          dim4 fill_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, align_w - score_tile_4.shape.w};
          int elem_size = 4;
          int offset = score_tile_4.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_4.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_4.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_2_stride;
    tpu_aligned_stride(&scores_max_2_stride, 0, &scores_max_2.shape, DT_FP32);
    scores_max_2_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_4.addr, score_tile_4.addr, scores_max_2.addr, &score_tile_4.shape, (score_tile_4.default_stride ? NULL : &score_tile_4.stride), (score_tile_4.default_stride ? NULL : &score_tile_4.stride), &scores_max_2_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_2.addr);
    tpu_bdc_load_fp32_exp_table(table_2.addr);
    tpu_bdc_fp32_exp(score_tile_4.addr, score_tile_4.addr, work_score0_2.addr, work_score1_2.addr, coeff_2.addr, table_2.addr, &score_tile_4.shape);
    __ppl_tensor_info tmp_buffer_sum_2 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_2.stride, 0, &tmp_buffer_sum_2.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_4.shape.n, score_tile_4.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_4.shape.n, score_tile_4.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_4.shape.n, score_tile_4.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_4.shape.n, score_tile_4.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_4.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, score_tile_4.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_4.shape.w) {
          dim4 fill_shape = {score_tile_4.shape.n, score_tile_4.shape.c, 1, align_w - score_tile_4.shape.w};
          int elem_size = 4;
          int offset = score_tile_4.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_4.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_4.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_5 = {.shape = {1, 1, 1, 192} , .stride = score_tile_4.stride, .addr = score_tile_4.addr + ((0) * score_tile_4.stride.c+(0) * score_tile_4.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_4.stride.c+(0) * score_tile_4.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_4.default_stride};
    __ppl_tensor_info prob_tile_5 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_4.stride, .addr = prob_tile_4.addr + ((0) * prob_tile_4.stride.c+(0) * prob_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_4.stride.c+(0) * prob_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_4.default_stride};
    tpu_bdc_cast(prob_tile_5.addr, score_tile_5.addr, &prob_tile_5.shape, (prob_tile_5.default_stride ? NULL : &prob_tile_5.stride), (score_tile_5.default_stride ? NULL : &score_tile_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_4_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_4.addr, ctx_acc_4_scalar_f32, &ctx_acc_4.shape, (ctx_acc_4.default_stride ? NULL : &ctx_acc_4.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_4.addr, prob_tile_4.addr, kv_cache_full_4.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_2_stride;
    tpu_aligned_stride(&scores_sum_2_stride, 0, &scores_sum_2.shape, DT_FP32);
    scores_sum_2_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_4.addr, ctx_acc_4.addr, scores_sum_2.addr, &ctx_acc_4.shape, (ctx_acc_4.default_stride ? NULL : &ctx_acc_4.stride), (ctx_acc_4.default_stride ? NULL : &ctx_acc_4.stride), &scores_sum_2_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_5 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_4.stride, .addr = ctx_acc_4.addr + ((0) * ctx_acc_4.stride.c+(0) * ctx_acc_4.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_4.stride.c+(0) * ctx_acc_4.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_4.default_stride};
    __ppl_tensor_info ctx_head_7 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_6.stride, .addr = ctx_head_6.addr + ((0) * ctx_head_6.stride.c+(0) * ctx_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_6.stride.c+(0) * ctx_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_6.default_stride};
    tpu_bdc_cast(ctx_head_7.addr, ctx_acc_5.addr, &ctx_head_7.shape, (ctx_head_7.default_stride ? NULL : &ctx_head_7.stride), (ctx_acc_5.default_stride ? NULL : &ctx_acc_5.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_6.addr, out_part_6_scalar_f32, &out_part_6.shape, (out_part_6.default_stride ? NULL : &out_part_6.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_8 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_6.stride, .addr = ctx_head_6.addr + (((0) / 64) * ctx_head_6.stride.c+(0) * ctx_head_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_6.stride.c+(0) * ctx_head_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_9 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_6.stride, .addr = wukv_full_tile_6.addr + (((128) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_7 = {.shape = {1, 1, 1, 128}, .stride = out_part_6.stride, .addr = out_part_6.addr + (((0) / 64) * out_part_6.stride.c+(0) * out_part_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_6.stride.c+(0) * out_part_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_7.addr, ctx_head_8.addr, wukv_full_tile_9.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_8 = {.shape = {1, 1, 1, 128} , .stride = out_part_6.stride, .addr = out_part_6.addr + ((0) * out_part_6.stride.c+(0) * out_part_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_6.stride.c+(0) * out_part_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_6.default_stride};
    __ppl_tensor_info out_cast_7 = {.shape = {1, 1, 1, 128} , .stride = out_cast_6.stride, .addr = out_cast_6.addr + ((0) * out_cast_6.stride.c+(0) * out_cast_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_6.stride.c+(0) * out_cast_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_6.default_stride};
    tpu_bdc_cast(out_cast_7.addr, out_part_8.addr, &out_cast_7.shape, (out_cast_7.default_stride ? NULL : &out_cast_7.stride), (out_part_8.default_stride ? NULL : &out_part_8.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_8 = {.shape = {1, 1, 1, 128} , .stride = out_cast_6.stride, .addr = out_cast_6.addr + ((0) * out_cast_6.stride.c+(0) * out_cast_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_6.stride.c+(0) * out_cast_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_6.default_stride};
    __ppl_tensor_info OUT_2 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(256) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(256) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_2.addr, out_cast_8.addr, &OUT_2.shape, (OUT_2.default_stride ? NULL : &OUT_2.stride), (out_cast_8.default_stride ? NULL : &out_cast_8.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_9 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_9.stride, 0, &q_nope_head_9.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_9 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_9.stride, 0, &q_wukv_acc_9.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_9 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_9.stride, 0, &q_wukv_head_9.shape, DT_BFP16);
    __ppl_tensor_info score_tile_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_6.stride, 0, &score_tile_6.shape, DT_FP32);
    __ppl_tensor_info prob_tile_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_6.stride, 0, &prob_tile_6.shape, DT_BFP16);
    __ppl_tensor_info scores_max_3 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_3.stride, 0, &scores_max_3.shape, DT_FP32);
    __ppl_tensor_info scores_sum_3 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_3.stride, 0, &scores_sum_3.shape, DT_FP32);
    __ppl_tensor_info work_score0_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_3.stride, 0, &work_score0_3.shape, DT_FP32);
    __ppl_tensor_info work_score1_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_3.stride, 0, &work_score1_3.shape, DT_FP32);
    __ppl_tensor_info coeff_3 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_3.stride, 0, &coeff_3.shape, DT_FP32);
    __ppl_tensor_info table_3 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_3.stride, 0, &table_3.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_6 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_6.stride, 0, &ctx_acc_6.shape, DT_FP32);
    __ppl_tensor_info ctx_head_9 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_9.stride, 0, &ctx_head_9.shape, DT_BFP16);
    __ppl_tensor_info out_part_9 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_9.stride, 0, &out_part_9.shape, DT_FP32);
    __ppl_tensor_info out_cast_9 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_9.stride, 0, &out_cast_9.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_7 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_4.stride, .addr = q_upper_h1_4.addr + ((0) * q_upper_h1_4.stride.c+(0) * q_upper_h1_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_4.stride.c+(0) * q_upper_h1_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_4.default_stride};
    __ppl_tensor_info q_nope_head_10 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_9.stride, .addr = q_nope_head_9.addr + ((0) * q_nope_head_9.stride.c+(0) * q_nope_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_9.stride.c+(0) * q_nope_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_9.default_stride};
    tpu_bdc_cpy(q_nope_head_10.addr, q_upper_h1_7.addr, &q_nope_head_10.shape, (q_nope_head_10.default_stride ? NULL : &q_nope_head_10.stride), (q_upper_h1_7.default_stride ? NULL : &q_upper_h1_7.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_9.addr, q_wukv_acc_9_scalar_f32, &q_wukv_acc_9.shape, (q_wukv_acc_9.default_stride ? NULL : &q_wukv_acc_9.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_11 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_9.stride, .addr = q_nope_head_9.addr + (((0) / 64) * q_nope_head_9.stride.c+(0) * q_nope_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_9.stride.c+(0) * q_nope_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_10 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_6.stride, .addr = wukv_full_tile_6.addr + (((256) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_10 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_9.stride, .addr = q_wukv_acc_9.addr + (((0) / 64) * q_wukv_acc_9.stride.c+(0) * q_wukv_acc_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_9.stride.c+(0) * q_wukv_acc_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_10.addr, q_nope_head_11.addr, wukv_full_tile_10.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_11 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_9.stride, .addr = q_wukv_acc_9.addr + ((0) * q_wukv_acc_9.stride.c+(0) * q_wukv_acc_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_9.stride.c+(0) * q_wukv_acc_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_9.default_stride};
    __ppl_tensor_info q_wukv_head_10 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_9.stride, .addr = q_wukv_head_9.addr + ((0) * q_wukv_head_9.stride.c+(0) * q_wukv_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_9.stride.c+(0) * q_wukv_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_9.default_stride};
    tpu_bdc_cast(q_wukv_head_10.addr, q_wukv_acc_11.addr, &q_wukv_head_10.shape, (q_wukv_head_10.default_stride ? NULL : &q_wukv_head_10.stride), (q_wukv_acc_11.default_stride ? NULL : &q_wukv_acc_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_11 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_9.stride, .addr = q_wukv_head_9.addr + (((0) / 64) * q_wukv_head_9.stride.c+(0) * q_wukv_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_9.stride.c+(0) * q_wukv_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_7 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_4.stride, .addr = kv_cache_full_4.addr + (((0) / 64) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_4.stride.c+(0) * kv_cache_full_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_5 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_3.stride, .addr = score_latent_h0_3.addr + (((0) / 64) * score_latent_h0_3.stride.c+(0) * score_latent_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_3.stride.c+(0) * score_latent_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_5.addr, q_wukv_head_11.addr, kv_cache_full_7.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_3 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_2.stride, .addr = q_rope_h1_2.addr + (((0) / 64) * q_rope_h1_2.stride.c+(0) * q_rope_h1_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_2.stride.c+(0) * q_rope_h1_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_7 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_4.stride, .addr = pe_score_tile_4.addr + (((0) / 64) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_4.stride.c+(0) * pe_score_tile_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_5 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_3.stride, .addr = score_rope_h0_3.addr + (((0) / 64) * score_rope_h0_3.stride.c+(0) * score_rope_h0_3.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_3.stride.c+(0) * score_rope_h0_3.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_5.addr, q_rope_h1_3.addr, pe_score_tile_7.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_6.addr, score_latent_h0_3.addr, score_rope_h0_3.addr, &score_tile_6.shape, (score_tile_6.default_stride ? NULL : &score_tile_6.stride), (score_latent_h0_3.default_stride ? NULL : &score_latent_h0_3.stride), (score_rope_h0_3.default_stride ? NULL : &score_rope_h0_3.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_6.addr, score_tile_6.addr, mask_row_2.addr, &score_tile_6.shape, (score_tile_6.default_stride ? NULL : &score_tile_6.stride), (score_tile_6.default_stride ? NULL : &score_tile_6.stride), (mask_row_2.default_stride ? NULL : &mask_row_2.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_6.addr, score_tile_6.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_6.shape, (score_tile_6.default_stride ? NULL : &score_tile_6.stride), (score_tile_6.default_stride ? NULL : &score_tile_6.stride), DT_FP32);
    {
    scalar_t scores_max_3_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_3.addr, scores_max_3_scalar_f32, &scores_max_3.shape, (scores_max_3.default_stride ? NULL : &scores_max_3.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_3 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_3.stride, 0, &tmp_buffer_max_3.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_6.shape.n, score_tile_6.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_6.shape.n, score_tile_6.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_6.shape.n, score_tile_6.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_6.shape.n, score_tile_6.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_6.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, score_tile_6.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_6.shape.w) {
          dim4 fill_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, align_w - score_tile_6.shape.w};
          int elem_size = 4;
          int offset = score_tile_6.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_6.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_6.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_3_stride;
    tpu_aligned_stride(&scores_max_3_stride, 0, &scores_max_3.shape, DT_FP32);
    scores_max_3_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_6.addr, score_tile_6.addr, scores_max_3.addr, &score_tile_6.shape, (score_tile_6.default_stride ? NULL : &score_tile_6.stride), (score_tile_6.default_stride ? NULL : &score_tile_6.stride), &scores_max_3_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_3.addr);
    tpu_bdc_load_fp32_exp_table(table_3.addr);
    tpu_bdc_fp32_exp(score_tile_6.addr, score_tile_6.addr, work_score0_3.addr, work_score1_3.addr, coeff_3.addr, table_3.addr, &score_tile_6.shape);
    __ppl_tensor_info tmp_buffer_sum_3 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_3.stride, 0, &tmp_buffer_sum_3.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_6.shape.n, score_tile_6.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_6.shape.n, score_tile_6.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_6.shape.n, score_tile_6.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_6.shape.n, score_tile_6.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_6.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, score_tile_6.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_6.shape.w) {
          dim4 fill_shape = {score_tile_6.shape.n, score_tile_6.shape.c, 1, align_w - score_tile_6.shape.w};
          int elem_size = 4;
          int offset = score_tile_6.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_6.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_6.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_3.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_7 = {.shape = {1, 1, 1, 192} , .stride = score_tile_6.stride, .addr = score_tile_6.addr + ((0) * score_tile_6.stride.c+(0) * score_tile_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_6.stride.c+(0) * score_tile_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_6.default_stride};
    __ppl_tensor_info prob_tile_7 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_6.stride, .addr = prob_tile_6.addr + ((0) * prob_tile_6.stride.c+(0) * prob_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_6.stride.c+(0) * prob_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_6.default_stride};
    tpu_bdc_cast(prob_tile_7.addr, score_tile_7.addr, &prob_tile_7.shape, (prob_tile_7.default_stride ? NULL : &prob_tile_7.stride), (score_tile_7.default_stride ? NULL : &score_tile_7.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_6.addr, ctx_acc_6_scalar_f32, &ctx_acc_6.shape, (ctx_acc_6.default_stride ? NULL : &ctx_acc_6.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_6.addr, prob_tile_6.addr, kv_cache_full_4.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_3_stride;
    tpu_aligned_stride(&scores_sum_3_stride, 0, &scores_sum_3.shape, DT_FP32);
    scores_sum_3_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_6.addr, ctx_acc_6.addr, scores_sum_3.addr, &ctx_acc_6.shape, (ctx_acc_6.default_stride ? NULL : &ctx_acc_6.stride), (ctx_acc_6.default_stride ? NULL : &ctx_acc_6.stride), &scores_sum_3_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_7 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_6.stride, .addr = ctx_acc_6.addr + ((0) * ctx_acc_6.stride.c+(0) * ctx_acc_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_6.stride.c+(0) * ctx_acc_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_6.default_stride};
    __ppl_tensor_info ctx_head_10 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_9.stride, .addr = ctx_head_9.addr + ((0) * ctx_head_9.stride.c+(0) * ctx_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_9.stride.c+(0) * ctx_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_9.default_stride};
    tpu_bdc_cast(ctx_head_10.addr, ctx_acc_7.addr, &ctx_head_10.shape, (ctx_head_10.default_stride ? NULL : &ctx_head_10.stride), (ctx_acc_7.default_stride ? NULL : &ctx_acc_7.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_9.addr, out_part_9_scalar_f32, &out_part_9.shape, (out_part_9.default_stride ? NULL : &out_part_9.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_11 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_9.stride, .addr = ctx_head_9.addr + (((0) / 64) * ctx_head_9.stride.c+(0) * ctx_head_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_9.stride.c+(0) * ctx_head_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_11 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_6.stride, .addr = wukv_full_tile_6.addr + (((384) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_6.stride.c+(0) * wukv_full_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_10 = {.shape = {1, 1, 1, 128}, .stride = out_part_9.stride, .addr = out_part_9.addr + (((0) / 64) * out_part_9.stride.c+(0) * out_part_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_9.stride.c+(0) * out_part_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_10.addr, ctx_head_11.addr, wukv_full_tile_11.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_11 = {.shape = {1, 1, 1, 128} , .stride = out_part_9.stride, .addr = out_part_9.addr + ((0) * out_part_9.stride.c+(0) * out_part_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_9.stride.c+(0) * out_part_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_9.default_stride};
    __ppl_tensor_info out_cast_10 = {.shape = {1, 1, 1, 128} , .stride = out_cast_9.stride, .addr = out_cast_9.addr + ((0) * out_cast_9.stride.c+(0) * out_cast_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_9.stride.c+(0) * out_cast_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_9.default_stride};
    tpu_bdc_cast(out_cast_10.addr, out_part_11.addr, &out_cast_10.shape, (out_cast_10.default_stride ? NULL : &out_cast_10.stride), (out_part_11.default_stride ? NULL : &out_part_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_11 = {.shape = {1, 1, 1, 128} , .stride = out_cast_9.stride, .addr = out_cast_9.addr + ((0) * out_cast_9.stride.c+(0) * out_cast_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_9.stride.c+(0) * out_cast_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_9.default_stride};
    __ppl_tensor_info OUT_3 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(384) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(384) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_3.addr, out_cast_11.addr, &OUT_3.shape, (OUT_3.default_stride ? NULL : &OUT_3.stride), (out_cast_11.default_stride ? NULL : &out_cast_11.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_6.stride, 0, &q_upper_acc_h0_6.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_8.stride, 0, &q_upper_h0_8.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_6.stride, 0, &q_upper_acc_h1_6.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_8.stride, 0, &q_upper_h1_8.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_4.stride, 0, &q_rope_src_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_4.stride, 0, &q_rope_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_2.stride, 0, &q_rope_tmp0_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_2.stride, 0, &q_rope_tmp1_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_2.stride, 0, &q_rope_tmp2_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_2.stride, 0, &q_rope_tmp3_h0_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_4.stride, 0, &q_rope_src_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_4.stride, 0, &q_rope_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_2.stride, 0, &q_rope_tmp0_h1_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_2.stride, 0, &q_rope_tmp1_h1_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_2.stride, 0, &q_rope_tmp2_h1_2.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_2.stride, 0, &q_rope_tmp3_h1_2.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_4.stride, 0, &cos_shared_4.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_4.stride, 0, &sin_shared_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_4.stride, 0, &pe_cur_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_2.stride, 0, &pe_cur_cos_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_2.stride, 0, &pe_cur_sin_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_2.stride, 0, &pe_cur_neg_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_2 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_2.stride, 0, &pe_cur_neg_sin_2.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_4.stride, 0, &pe_cur_rope_4.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_12 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_12.stride, 0, &wukv_full_tile_12.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_8 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_8.stride, 0, &kv_cache_full_8.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_8 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_8.stride, 0, &pe_score_tile_8.shape, DT_BFP16);
  __ppl_tensor_info mask_row_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_4.stride, 0, &mask_row_4.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_6.stride, 0, &score_latent_h0_6.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_6.stride, 0, &score_rope_h0_6.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_2.stride, 0, &score_latent_h1_2.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_2 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_2.stride, 0, &score_rope_h1_2.shape, DT_FP32);
  if (tpu_workitem_index() == 2) {
    __ppl_tensor_info COS_2 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_5 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_4.stride, .addr = cos_shared_4.addr + ((0) * cos_shared_4.stride.c+(0) * cos_shared_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_4.stride.c+(0) * cos_shared_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_4.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_5.addr, COS_2.addr, &cos_shared_5.shape, (cos_shared_5.default_stride ? NULL : &cos_shared_5.stride), (COS_2.default_stride ? NULL : &COS_2.stride), DT_BFP16);
    __ppl_tensor_info SIN_2 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_5 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_4.stride, .addr = sin_shared_4.addr + ((0) * sin_shared_4.stride.c+(0) * sin_shared_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_4.stride.c+(0) * sin_shared_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_4.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_5.addr, SIN_2.addr, &sin_shared_5.shape, (sin_shared_5.default_stride ? NULL : &sin_shared_5.stride), (SIN_2.default_stride ? NULL : &SIN_2.stride), DT_BFP16);
    for (int d_4 = 0; d_4 < 4; ++d_4) {
      __ppl_tensor_info kv_cur_tile_6 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_6.stride, 0, &kv_cur_tile_6.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_2 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_4 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_4 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_7 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_6.stride, .addr = kv_cur_tile_6.addr + ((0) * kv_cur_tile_6.stride.c+(0) * kv_cur_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_6.stride.c+(0) * kv_cur_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_6.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_7.addr, KV_CUR_2.addr, &kv_cur_tile_7.shape, (kv_cur_tile_7.default_stride ? NULL : &kv_cur_tile_7.stride), (KV_CUR_2.default_stride ? NULL : &KV_CUR_2.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_8 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_6.stride, .addr = kv_cur_tile_6.addr + ((0) * kv_cur_tile_6.stride.c+(0) * kv_cur_tile_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_6.stride.c+(0) * kv_cur_tile_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_6.default_stride};
      __ppl_tensor_info KVcache_OUT_4 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_4 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_4 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_4.addr, kv_cur_tile_8.addr, &KVcache_OUT_4.shape, (KVcache_OUT_4.default_stride ? NULL : &KVcache_OUT_4.stride), (kv_cur_tile_8.default_stride ? NULL : &kv_cur_tile_8.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_2 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_5 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_4.stride, .addr = pe_cur_4.addr + ((0) * pe_cur_4.stride.c+(0) * pe_cur_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_4.stride.c+(0) * pe_cur_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_4.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_5.addr, PE_CUR_2.addr, &pe_cur_5.shape, (pe_cur_5.default_stride ? NULL : &pe_cur_5.stride), (PE_CUR_2.default_stride ? NULL : &PE_CUR_2.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_2.addr, pe_cur_4.addr, cos_shared_4.addr, &pe_cur_cos_2.shape, (pe_cur_cos_2.default_stride ? NULL : &pe_cur_cos_2.stride), (pe_cur_4.default_stride ? NULL : &pe_cur_4.stride), (cos_shared_4.default_stride ? NULL : &cos_shared_4.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_2.addr, pe_cur_4.addr, sin_shared_4.addr, &pe_cur_sin_2.shape, (pe_cur_sin_2.default_stride ? NULL : &pe_cur_sin_2.stride), (pe_cur_4.default_stride ? NULL : &pe_cur_4.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_2_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_2_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_2_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_2.addr, pe_cur_4.addr, pe_cur_neg_2_scalar_DT_BFP16, &pe_cur_neg_2.shape, (pe_cur_neg_2.default_stride ? NULL : &pe_cur_neg_2.stride), (pe_cur_4.default_stride ? NULL : &pe_cur_4.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_2.addr, pe_cur_neg_2.addr, sin_shared_4.addr, &pe_cur_neg_sin_2.shape, (pe_cur_neg_sin_2.default_stride ? NULL : &pe_cur_neg_sin_2.stride), (pe_cur_neg_2.default_stride ? NULL : &pe_cur_neg_2.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_4.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_4.shape.n, .c = pe_cur_rope_4.shape.c, .h = pe_cur_rope_4.shape.h, .w = pe_cur_rope_4.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_4.addr, pe_cur_cos_2.addr, pe_cur_neg_sin_2.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_4.addr + 2, pe_cur_cos_2.addr + 2, pe_cur_sin_2.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_5 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_4.stride, .addr = pe_cur_rope_4.addr + ((0) * pe_cur_rope_4.stride.c+(0) * pe_cur_rope_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_4.stride.c+(0) * pe_cur_rope_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_4.default_stride};
    __ppl_tensor_info PEcache_OUT_4 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_4.addr, pe_cur_rope_5.addr, &PEcache_OUT_4.shape, (PEcache_OUT_4.default_stride ? NULL : &PEcache_OUT_4.stride), (pe_cur_rope_5.default_stride ? NULL : &pe_cur_rope_5.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_6.addr, q_upper_acc_h0_6_scalar_f32, &q_upper_acc_h0_6.shape, (q_upper_acc_h0_6.default_stride ? NULL : &q_upper_acc_h0_6.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_6.addr, q_upper_acc_h1_6_scalar_f32, &q_upper_acc_h1_6.shape, (q_upper_acc_h1_6.default_stride ? NULL : &q_upper_acc_h1_6.stride), DT_FP32);
    }
    for (int k_2 = 0; k_2 < 12; ++k_2) {
      __ppl_tensor_info q_tile_8 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_8.stride, 0, &q_tile_8.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_6 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_6.stride, 0, &wuq_fp8_6.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_16 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_16.stride, 0, &wuq_scale_stage_16.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_8 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_8.stride, 0, &wuq_tile_8.shape, DT_BFP16);
      __ppl_tensor_info Q_2 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_2 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_2 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_9 = {.shape = {1, 1, 1, 128} , .stride = q_tile_8.stride, .addr = q_tile_8.addr + ((0) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_8.default_stride};
      tpu_gdma_cpy_S2L(q_tile_9.addr, Q_2.addr, &q_tile_9.shape, (q_tile_9.default_stride ? NULL : &q_tile_9.stride), (Q_2.default_stride ? NULL : &Q_2.stride), DT_BFP16);
      __ppl_tensor_info WUQ_2 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((768) * 1536+((k_2 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((768) * 1536+((k_2 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_7 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_6.stride, .addr = wuq_fp8_6.addr + ((0) * wuq_fp8_6.stride.c+(0) * wuq_fp8_6.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_6.stride.c+(0) * wuq_fp8_6.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_6.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_7.addr, WUQ_2.addr, &wuq_fp8_7.shape, (wuq_fp8_7.default_stride ? NULL : &wuq_fp8_7.stride), (WUQ_2.default_stride ? NULL : &WUQ_2.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_36 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_36.stride, 0, &wuq_scale_scalar_36.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_12 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((6) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((6) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_37 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_36.stride, .addr = wuq_scale_scalar_36.addr + ((0) * wuq_scale_scalar_36.stride.c+(0) * wuq_scale_scalar_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_36.stride.c+(0) * wuq_scale_scalar_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_36.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_37.addr, WUQ_SCALE_12.addr, &wuq_scale_scalar_37.shape, (wuq_scale_scalar_37.default_stride ? NULL : &wuq_scale_scalar_37.stride), (WUQ_SCALE_12.default_stride ? NULL : &WUQ_SCALE_12.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_38 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_36.stride, .addr = wuq_scale_scalar_36.addr + (((0) / 64) * wuq_scale_scalar_36.stride.c+(0) * wuq_scale_scalar_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_36.stride.c+(0) * wuq_scale_scalar_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_17 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((0) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_17.addr, wuq_scale_scalar_38.addr, &wuq_scale_stage_17.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_39 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_39.stride, 0, &wuq_scale_scalar_39.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_13 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((6) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((6) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_40 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_39.stride, .addr = wuq_scale_scalar_39.addr + ((0) * wuq_scale_scalar_39.stride.c+(0) * wuq_scale_scalar_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_39.stride.c+(0) * wuq_scale_scalar_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_39.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_40.addr, WUQ_SCALE_13.addr, &wuq_scale_scalar_40.shape, (wuq_scale_scalar_40.default_stride ? NULL : &wuq_scale_scalar_40.stride), (WUQ_SCALE_13.default_stride ? NULL : &WUQ_SCALE_13.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_41 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_39.stride, .addr = wuq_scale_scalar_39.addr + (((0) / 64) * wuq_scale_scalar_39.stride.c+(0) * wuq_scale_scalar_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_39.stride.c+(0) * wuq_scale_scalar_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_18 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((64) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_18.addr, wuq_scale_scalar_41.addr, &wuq_scale_stage_18.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_42 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_42.stride, 0, &wuq_scale_scalar_42.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_14 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((7) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((7) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_43 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_42.stride, .addr = wuq_scale_scalar_42.addr + ((0) * wuq_scale_scalar_42.stride.c+(0) * wuq_scale_scalar_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_42.stride.c+(0) * wuq_scale_scalar_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_42.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_43.addr, WUQ_SCALE_14.addr, &wuq_scale_scalar_43.shape, (wuq_scale_scalar_43.default_stride ? NULL : &wuq_scale_scalar_43.stride), (WUQ_SCALE_14.default_stride ? NULL : &WUQ_SCALE_14.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_44 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_42.stride, .addr = wuq_scale_scalar_42.addr + (((0) / 64) * wuq_scale_scalar_42.stride.c+(0) * wuq_scale_scalar_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_42.stride.c+(0) * wuq_scale_scalar_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_19 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((128) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_19.addr, wuq_scale_scalar_44.addr, &wuq_scale_stage_19.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_45 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_45.stride, 0, &wuq_scale_scalar_45.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_15 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((7) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((7) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_46 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_45.stride, .addr = wuq_scale_scalar_45.addr + ((0) * wuq_scale_scalar_45.stride.c+(0) * wuq_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_45.stride.c+(0) * wuq_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_45.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_46.addr, WUQ_SCALE_15.addr, &wuq_scale_scalar_46.shape, (wuq_scale_scalar_46.default_stride ? NULL : &wuq_scale_scalar_46.stride), (WUQ_SCALE_15.default_stride ? NULL : &WUQ_SCALE_15.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_47 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_45.stride, .addr = wuq_scale_scalar_45.addr + (((0) / 64) * wuq_scale_scalar_45.stride.c+(0) * wuq_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_45.stride.c+(0) * wuq_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_20 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((192) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_20.addr, wuq_scale_scalar_47.addr, &wuq_scale_stage_20.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_48 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_48.stride, 0, &wuq_scale_scalar_48.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_16 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((8) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((8) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_49 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_48.stride, .addr = wuq_scale_scalar_48.addr + ((0) * wuq_scale_scalar_48.stride.c+(0) * wuq_scale_scalar_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_48.stride.c+(0) * wuq_scale_scalar_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_48.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_49.addr, WUQ_SCALE_16.addr, &wuq_scale_scalar_49.shape, (wuq_scale_scalar_49.default_stride ? NULL : &wuq_scale_scalar_49.stride), (WUQ_SCALE_16.default_stride ? NULL : &WUQ_SCALE_16.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_50 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_48.stride, .addr = wuq_scale_scalar_48.addr + (((0) / 64) * wuq_scale_scalar_48.stride.c+(0) * wuq_scale_scalar_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_48.stride.c+(0) * wuq_scale_scalar_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_21 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((256) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_21.addr, wuq_scale_scalar_50.addr, &wuq_scale_stage_21.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_51 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_51.stride, 0, &wuq_scale_scalar_51.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_17 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((8) * 12+(k_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((8) * 12+(k_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_52 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_51.stride, .addr = wuq_scale_scalar_51.addr + ((0) * wuq_scale_scalar_51.stride.c+(0) * wuq_scale_scalar_51.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_51.stride.c+(0) * wuq_scale_scalar_51.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_51.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_52.addr, WUQ_SCALE_17.addr, &wuq_scale_scalar_52.shape, (wuq_scale_scalar_52.default_stride ? NULL : &wuq_scale_scalar_52.stride), (WUQ_SCALE_17.default_stride ? NULL : &WUQ_SCALE_17.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_53 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_51.stride, .addr = wuq_scale_scalar_51.addr + (((0) / 64) * wuq_scale_scalar_51.stride.c+(0) * wuq_scale_scalar_51.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_51.stride.c+(0) * wuq_scale_scalar_51.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_22 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((320) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_22.addr, wuq_scale_scalar_53.addr, &wuq_scale_stage_22.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_8 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_6.stride, .addr = wuq_fp8_6.addr + (((0) / 64) * wuq_fp8_6.stride.c+(0) * wuq_fp8_6.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_6.stride.c+(0) * wuq_fp8_6.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_23 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_16.stride, .addr = wuq_scale_stage_16.addr + (((0) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_16.stride.c+(0) * wuq_scale_stage_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_9 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_8.stride, .addr = wuq_tile_8.addr + (((0) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_9.addr, wuq_fp8_8.addr, &wuq_tile_9.shape, (wuq_tile_9.default_stride ? NULL : &wuq_tile_9.stride), (wuq_fp8_8.default_stride ? NULL : &wuq_fp8_8.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_23_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_23_block_stride, 0, &wuq_scale_stage_23.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_23_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_9.addr, wuq_tile_9.addr, wuq_scale_stage_23.addr, &wuq_tile_9.shape, (wuq_tile_9.default_stride ? NULL : &wuq_tile_9.stride), (wuq_tile_9.default_stride ? NULL : &wuq_tile_9.stride), &wuq_scale_stage_23_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_10 = {.shape = {1, 1, 1, 128}, .stride = q_tile_8.stride, .addr = q_tile_8.addr + (((0) / 64) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_10 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_8.stride, .addr = wuq_tile_8.addr + (((0) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_7 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_6.stride, .addr = q_upper_acc_h0_6.addr + (((0) / 64) * q_upper_acc_h0_6.stride.c+(0) * q_upper_acc_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_6.stride.c+(0) * q_upper_acc_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_7.addr, q_tile_10.addr, wuq_tile_10.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_11 = {.shape = {1, 1, 1, 128}, .stride = q_tile_8.stride, .addr = q_tile_8.addr + (((0) / 64) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_8.stride.c+(0) * q_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_11 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_8.stride, .addr = wuq_tile_8.addr + (((192) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_8.stride.c+(0) * wuq_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_7 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_6.stride, .addr = q_upper_acc_h1_6.addr + (((0) / 64) * q_upper_acc_h1_6.stride.c+(0) * q_upper_acc_h1_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_6.stride.c+(0) * q_upper_acc_h1_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_7.addr, q_tile_11.addr, wuq_tile_11.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_8 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_6.stride, .addr = q_upper_acc_h0_6.addr + ((0) * q_upper_acc_h0_6.stride.c+(0) * q_upper_acc_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_6.stride.c+(0) * q_upper_acc_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_6.default_stride};
    __ppl_tensor_info q_upper_h0_9 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_8.stride, .addr = q_upper_h0_8.addr + ((0) * q_upper_h0_8.stride.c+(0) * q_upper_h0_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_8.stride.c+(0) * q_upper_h0_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_8.default_stride};
    tpu_bdc_cast(q_upper_h0_9.addr, q_upper_acc_h0_8.addr, &q_upper_h0_9.shape, (q_upper_h0_9.default_stride ? NULL : &q_upper_h0_9.stride), (q_upper_acc_h0_8.default_stride ? NULL : &q_upper_acc_h0_8.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_8 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_6.stride, .addr = q_upper_acc_h1_6.addr + ((0) * q_upper_acc_h1_6.stride.c+(0) * q_upper_acc_h1_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_6.stride.c+(0) * q_upper_acc_h1_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_6.default_stride};
    __ppl_tensor_info q_upper_h1_9 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_8.stride, .addr = q_upper_h1_8.addr + ((0) * q_upper_h1_8.stride.c+(0) * q_upper_h1_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_8.stride.c+(0) * q_upper_h1_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_8.default_stride};
    tpu_bdc_cast(q_upper_h1_9.addr, q_upper_acc_h1_8.addr, &q_upper_h1_9.shape, (q_upper_h1_9.default_stride ? NULL : &q_upper_h1_9.stride), (q_upper_acc_h1_8.default_stride ? NULL : &q_upper_acc_h1_8.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_5 = 0; d_5 < 4; ++d_5) {
      __ppl_tensor_info wukv_fp8_6 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_6.stride, 0, &wukv_fp8_6.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_12 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_12.stride, 0, &wukv_scale_stage_12.shape, DT_BFP16);
      __ppl_tensor_info WUKV_2 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((1024) * 512+((d_5 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((1024) * 512+((d_5 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_7 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_6.stride, .addr = wukv_fp8_6.addr + ((0) * wukv_fp8_6.stride.c+(0) * wukv_fp8_6.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_6.stride.c+(0) * wukv_fp8_6.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_6.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_7.addr, WUKV_2.addr, &wukv_fp8_7.shape, (wukv_fp8_7.default_stride ? NULL : &wukv_fp8_7.stride), (WUKV_2.default_stride ? NULL : &WUKV_2.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_20 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_20.stride, 0, &wukv_scale_scalar_20.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_8 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((8) * 4+(d_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((8) * 4+(d_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_21 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_20.stride, .addr = wukv_scale_scalar_20.addr + ((0) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_20.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_21.addr, WUKV_SCALE_8.addr, &wukv_scale_scalar_21.shape, (wukv_scale_scalar_21.default_stride ? NULL : &wukv_scale_scalar_21.stride), (WUKV_SCALE_8.default_stride ? NULL : &WUKV_SCALE_8.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_22 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_20.stride, .addr = wukv_scale_scalar_20.addr + (((0) / 64) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_13 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_12.stride, .addr = wukv_scale_stage_12.addr + (((0) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_13_chunk = wukv_scale_stage_13;
      wukv_scale_stage_13_chunk.shape.c = 64;
      wukv_scale_stage_13_chunk.addr = wukv_scale_stage_13.addr + ((0) * wukv_scale_stage_13.stride.c) * 2;
      wukv_scale_stage_13_chunk.offset = wukv_scale_stage_13.offset + ((0) * wukv_scale_stage_13.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_13_chunk.addr, wukv_scale_scalar_22.addr, &wukv_scale_stage_13_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_13_chunk_1 = wukv_scale_stage_13;
      wukv_scale_stage_13_chunk_1.shape.c = 64;
      wukv_scale_stage_13_chunk_1.addr = wukv_scale_stage_13.addr + ((1) * wukv_scale_stage_13.stride.c) * 2;
      wukv_scale_stage_13_chunk_1.offset = wukv_scale_stage_13.offset + ((1) * wukv_scale_stage_13.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_13_chunk_1.addr, wukv_scale_scalar_22.addr, &wukv_scale_stage_13_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_9 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((9) * 4+(d_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((9) * 4+(d_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_23 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_20.stride, .addr = wukv_scale_scalar_20.addr + ((0) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_20.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_23.addr, WUKV_SCALE_9.addr, &wukv_scale_scalar_23.shape, (wukv_scale_scalar_23.default_stride ? NULL : &wukv_scale_scalar_23.stride), (WUKV_SCALE_9.default_stride ? NULL : &WUKV_SCALE_9.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_24 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_20.stride, .addr = wukv_scale_scalar_20.addr + (((0) / 64) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_20.stride.c+(0) * wukv_scale_scalar_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_14 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_12.stride, .addr = wukv_scale_stage_12.addr + (((128) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_14_chunk = wukv_scale_stage_14;
      wukv_scale_stage_14_chunk.shape.c = 64;
      wukv_scale_stage_14_chunk.addr = wukv_scale_stage_14.addr + ((0) * wukv_scale_stage_14.stride.c) * 2;
      wukv_scale_stage_14_chunk.offset = wukv_scale_stage_14.offset + ((0) * wukv_scale_stage_14.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_14_chunk.addr, wukv_scale_scalar_24.addr, &wukv_scale_stage_14_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_14_chunk_1 = wukv_scale_stage_14;
      wukv_scale_stage_14_chunk_1.shape.c = 64;
      wukv_scale_stage_14_chunk_1.addr = wukv_scale_stage_14.addr + ((1) * wukv_scale_stage_14.stride.c) * 2;
      wukv_scale_stage_14_chunk_1.offset = wukv_scale_stage_14.offset + ((1) * wukv_scale_stage_14.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_14_chunk_1.addr, wukv_scale_scalar_24.addr, &wukv_scale_stage_14_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_25 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_25.stride, 0, &wukv_scale_scalar_25.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_10 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((10) * 4+(d_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((10) * 4+(d_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_26 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_25.stride, .addr = wukv_scale_scalar_25.addr + ((0) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_25.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_26.addr, WUKV_SCALE_10.addr, &wukv_scale_scalar_26.shape, (wukv_scale_scalar_26.default_stride ? NULL : &wukv_scale_scalar_26.stride), (WUKV_SCALE_10.default_stride ? NULL : &WUKV_SCALE_10.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_27 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_25.stride, .addr = wukv_scale_scalar_25.addr + (((0) / 64) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_15 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_12.stride, .addr = wukv_scale_stage_12.addr + (((256) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_15_chunk = wukv_scale_stage_15;
      wukv_scale_stage_15_chunk.shape.c = 64;
      wukv_scale_stage_15_chunk.addr = wukv_scale_stage_15.addr + ((0) * wukv_scale_stage_15.stride.c) * 2;
      wukv_scale_stage_15_chunk.offset = wukv_scale_stage_15.offset + ((0) * wukv_scale_stage_15.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_15_chunk.addr, wukv_scale_scalar_27.addr, &wukv_scale_stage_15_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_15_chunk_1 = wukv_scale_stage_15;
      wukv_scale_stage_15_chunk_1.shape.c = 64;
      wukv_scale_stage_15_chunk_1.addr = wukv_scale_stage_15.addr + ((1) * wukv_scale_stage_15.stride.c) * 2;
      wukv_scale_stage_15_chunk_1.offset = wukv_scale_stage_15.offset + ((1) * wukv_scale_stage_15.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_15_chunk_1.addr, wukv_scale_scalar_27.addr, &wukv_scale_stage_15_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_11 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((11) * 4+(d_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((11) * 4+(d_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_28 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_25.stride, .addr = wukv_scale_scalar_25.addr + ((0) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_25.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_28.addr, WUKV_SCALE_11.addr, &wukv_scale_scalar_28.shape, (wukv_scale_scalar_28.default_stride ? NULL : &wukv_scale_scalar_28.stride), (WUKV_SCALE_11.default_stride ? NULL : &WUKV_SCALE_11.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_29 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_25.stride, .addr = wukv_scale_scalar_25.addr + (((0) / 64) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_25.stride.c+(0) * wukv_scale_scalar_25.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_16 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_12.stride, .addr = wukv_scale_stage_12.addr + (((384) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_16_chunk = wukv_scale_stage_16;
      wukv_scale_stage_16_chunk.shape.c = 64;
      wukv_scale_stage_16_chunk.addr = wukv_scale_stage_16.addr + ((0) * wukv_scale_stage_16.stride.c) * 2;
      wukv_scale_stage_16_chunk.offset = wukv_scale_stage_16.offset + ((0) * wukv_scale_stage_16.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_16_chunk.addr, wukv_scale_scalar_29.addr, &wukv_scale_stage_16_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_16_chunk_1 = wukv_scale_stage_16;
      wukv_scale_stage_16_chunk_1.shape.c = 64;
      wukv_scale_stage_16_chunk_1.addr = wukv_scale_stage_16.addr + ((1) * wukv_scale_stage_16.stride.c) * 2;
      wukv_scale_stage_16_chunk_1.offset = wukv_scale_stage_16.offset + ((1) * wukv_scale_stage_16.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_16_chunk_1.addr, wukv_scale_scalar_29.addr, &wukv_scale_stage_16_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_8 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_6.stride, .addr = wukv_fp8_6.addr + (((0) / 64) * wukv_fp8_6.stride.c+(0) * wukv_fp8_6.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_6.stride.c+(0) * wukv_fp8_6.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_17 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_12.stride, .addr = wukv_scale_stage_12.addr + (((0) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_12.stride.c+(0) * wukv_scale_stage_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_13 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_12.stride, .addr = wukv_full_tile_12.addr + (((0) / 64) * wukv_full_tile_12.stride.c+((d_5 * 128)) * wukv_full_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_12.stride.c+((d_5 * 128)) * wukv_full_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_13.addr, wukv_fp8_8.addr, &wukv_full_tile_13.shape, (wukv_full_tile_13.default_stride ? NULL : &wukv_full_tile_13.stride), (wukv_fp8_8.default_stride ? NULL : &wukv_fp8_8.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_17_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_17_block_stride, 0, &wukv_scale_stage_17.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_17_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_13.addr, wukv_full_tile_13.addr, wukv_scale_stage_17.addr, &wukv_full_tile_13.shape, (wukv_full_tile_13.default_stride ? NULL : &wukv_full_tile_13.stride), (wukv_full_tile_13.default_stride ? NULL : &wukv_full_tile_13.stride), &wukv_scale_stage_17_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_5 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_9 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_8.stride, .addr = kv_cache_full_8.addr + ((0) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_8.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_9.addr, KVcache_OUT_5.addr, &kv_cache_full_9.shape, (kv_cache_full_9.default_stride ? NULL : &kv_cache_full_9.stride), (KVcache_OUT_5.default_stride ? NULL : &KVcache_OUT_5.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_5 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_9 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_8.stride, .addr = pe_score_tile_8.addr + ((0) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_8.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_9.addr, PEcache_OUT_5.addr, &pe_score_tile_9.shape, (pe_score_tile_9.default_stride ? NULL : &pe_score_tile_9.stride), (PEcache_OUT_5.default_stride ? NULL : &PEcache_OUT_5.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_2 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_5 = {.shape = {1, 1, 1, 192} , .stride = mask_row_4.stride, .addr = mask_row_4.addr + ((0) * mask_row_4.stride.c+(0) * mask_row_4.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_4.stride.c+(0) * mask_row_4.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_4.default_stride};
    tpu_gdma_cpy_S2L(mask_row_5.addr, DECODE_MASK_2.addr, &mask_row_5.shape, (mask_row_5.default_stride ? NULL : &mask_row_5.stride), (DECODE_MASK_2.default_stride ? NULL : &DECODE_MASK_2.stride), DT_FP32);
    {
    scalar_t score_latent_h0_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_6.addr, score_latent_h0_6_scalar_f32, &score_latent_h0_6.shape, (score_latent_h0_6.default_stride ? NULL : &score_latent_h0_6.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_6.addr, score_rope_h0_6_scalar_f32, &score_rope_h0_6.shape, (score_rope_h0_6.default_stride ? NULL : &score_rope_h0_6.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_2_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_2.addr, score_latent_h1_2_scalar_f32, &score_latent_h1_2.shape, (score_latent_h1_2.default_stride ? NULL : &score_latent_h1_2.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_2_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_2.addr, score_rope_h1_2_scalar_f32, &score_rope_h1_2.shape, (score_rope_h1_2.default_stride ? NULL : &score_rope_h1_2.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_10 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_8.stride, .addr = q_upper_h0_8.addr + ((0) * q_upper_h0_8.stride.c+(128) * q_upper_h0_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_8.stride.c+(128) * q_upper_h0_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_8.default_stride};
    __ppl_tensor_info q_rope_src_h0_5 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_4.stride, .addr = q_rope_src_h0_4.addr + ((0) * q_rope_src_h0_4.stride.c+(0) * q_rope_src_h0_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_4.stride.c+(0) * q_rope_src_h0_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_4.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_5.addr, q_upper_h0_10.addr, &q_rope_src_h0_5.shape, (q_rope_src_h0_5.default_stride ? NULL : &q_rope_src_h0_5.stride), (q_upper_h0_10.default_stride ? NULL : &q_upper_h0_10.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_2.addr, q_rope_src_h0_4.addr, cos_shared_4.addr, &q_rope_tmp0_h0_2.shape, (q_rope_tmp0_h0_2.default_stride ? NULL : &q_rope_tmp0_h0_2.stride), (q_rope_src_h0_4.default_stride ? NULL : &q_rope_src_h0_4.stride), (cos_shared_4.default_stride ? NULL : &cos_shared_4.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_2.addr, q_rope_src_h0_4.addr, sin_shared_4.addr, &q_rope_tmp1_h0_2.shape, (q_rope_tmp1_h0_2.default_stride ? NULL : &q_rope_tmp1_h0_2.stride), (q_rope_src_h0_4.default_stride ? NULL : &q_rope_src_h0_4.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_2_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_2_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_2_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_2.addr, q_rope_src_h0_4.addr, q_rope_tmp2_h0_2_scalar_DT_BFP16, &q_rope_tmp2_h0_2.shape, (q_rope_tmp2_h0_2.default_stride ? NULL : &q_rope_tmp2_h0_2.stride), (q_rope_src_h0_4.default_stride ? NULL : &q_rope_src_h0_4.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_2.addr, q_rope_tmp2_h0_2.addr, sin_shared_4.addr, &q_rope_tmp3_h0_2.shape, (q_rope_tmp3_h0_2.default_stride ? NULL : &q_rope_tmp3_h0_2.stride), (q_rope_tmp2_h0_2.default_stride ? NULL : &q_rope_tmp2_h0_2.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_4.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_4.shape.n, .c = q_rope_h0_4.shape.c, .h = q_rope_h0_4.shape.h, .w = q_rope_h0_4.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_4.addr, q_rope_tmp0_h0_2.addr, q_rope_tmp3_h0_2.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_4.addr + 2, q_rope_tmp0_h0_2.addr + 2, q_rope_tmp1_h0_2.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_10 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_8.stride, .addr = q_upper_h1_8.addr + ((0) * q_upper_h1_8.stride.c+(128) * q_upper_h1_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_8.stride.c+(128) * q_upper_h1_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_8.default_stride};
    __ppl_tensor_info q_rope_src_h1_5 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_4.stride, .addr = q_rope_src_h1_4.addr + ((0) * q_rope_src_h1_4.stride.c+(0) * q_rope_src_h1_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_4.stride.c+(0) * q_rope_src_h1_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_4.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_5.addr, q_upper_h1_10.addr, &q_rope_src_h1_5.shape, (q_rope_src_h1_5.default_stride ? NULL : &q_rope_src_h1_5.stride), (q_upper_h1_10.default_stride ? NULL : &q_upper_h1_10.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_2.addr, q_rope_src_h1_4.addr, cos_shared_4.addr, &q_rope_tmp0_h1_2.shape, (q_rope_tmp0_h1_2.default_stride ? NULL : &q_rope_tmp0_h1_2.stride), (q_rope_src_h1_4.default_stride ? NULL : &q_rope_src_h1_4.stride), (cos_shared_4.default_stride ? NULL : &cos_shared_4.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_2.addr, q_rope_src_h1_4.addr, sin_shared_4.addr, &q_rope_tmp1_h1_2.shape, (q_rope_tmp1_h1_2.default_stride ? NULL : &q_rope_tmp1_h1_2.stride), (q_rope_src_h1_4.default_stride ? NULL : &q_rope_src_h1_4.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_2_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_2_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_2_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_2.addr, q_rope_src_h1_4.addr, q_rope_tmp2_h1_2_scalar_DT_BFP16, &q_rope_tmp2_h1_2.shape, (q_rope_tmp2_h1_2.default_stride ? NULL : &q_rope_tmp2_h1_2.stride), (q_rope_src_h1_4.default_stride ? NULL : &q_rope_src_h1_4.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_2.addr, q_rope_tmp2_h1_2.addr, sin_shared_4.addr, &q_rope_tmp3_h1_2.shape, (q_rope_tmp3_h1_2.default_stride ? NULL : &q_rope_tmp3_h1_2.stride), (q_rope_tmp2_h1_2.default_stride ? NULL : &q_rope_tmp2_h1_2.stride), (sin_shared_4.default_stride ? NULL : &sin_shared_4.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_4.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_4.shape.n, .c = q_rope_h1_4.shape.c, .h = q_rope_h1_4.shape.h, .w = q_rope_h1_4.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_4.addr, q_rope_tmp0_h1_2.addr, q_rope_tmp3_h1_2.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_4.addr + 2, q_rope_tmp0_h1_2.addr + 2, q_rope_tmp1_h1_2.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_12 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_12.stride, 0, &q_nope_head_12.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_12 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_12.stride, 0, &q_wukv_acc_12.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_12 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_12.stride, 0, &q_wukv_head_12.shape, DT_BFP16);
    __ppl_tensor_info score_tile_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_8.stride, 0, &score_tile_8.shape, DT_FP32);
    __ppl_tensor_info prob_tile_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_8.stride, 0, &prob_tile_8.shape, DT_BFP16);
    __ppl_tensor_info scores_max_4 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_4.stride, 0, &scores_max_4.shape, DT_FP32);
    __ppl_tensor_info scores_sum_4 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_4.stride, 0, &scores_sum_4.shape, DT_FP32);
    __ppl_tensor_info work_score0_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_4.stride, 0, &work_score0_4.shape, DT_FP32);
    __ppl_tensor_info work_score1_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_4.stride, 0, &work_score1_4.shape, DT_FP32);
    __ppl_tensor_info coeff_4 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_4.stride, 0, &coeff_4.shape, DT_FP32);
    __ppl_tensor_info table_4 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_4.stride, 0, &table_4.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_8 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_8.stride, 0, &ctx_acc_8.shape, DT_FP32);
    __ppl_tensor_info ctx_head_12 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_12.stride, 0, &ctx_head_12.shape, DT_BFP16);
    __ppl_tensor_info out_part_12 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_12.stride, 0, &out_part_12.shape, DT_FP32);
    __ppl_tensor_info out_cast_12 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_12.stride, 0, &out_cast_12.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_11 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_8.stride, .addr = q_upper_h0_8.addr + ((0) * q_upper_h0_8.stride.c+(0) * q_upper_h0_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_8.stride.c+(0) * q_upper_h0_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_8.default_stride};
    __ppl_tensor_info q_nope_head_13 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_12.stride, .addr = q_nope_head_12.addr + ((0) * q_nope_head_12.stride.c+(0) * q_nope_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_12.stride.c+(0) * q_nope_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_12.default_stride};
    tpu_bdc_cpy(q_nope_head_13.addr, q_upper_h0_11.addr, &q_nope_head_13.shape, (q_nope_head_13.default_stride ? NULL : &q_nope_head_13.stride), (q_upper_h0_11.default_stride ? NULL : &q_upper_h0_11.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_12.addr, q_wukv_acc_12_scalar_f32, &q_wukv_acc_12.shape, (q_wukv_acc_12.default_stride ? NULL : &q_wukv_acc_12.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_14 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_12.stride, .addr = q_nope_head_12.addr + (((0) / 64) * q_nope_head_12.stride.c+(0) * q_nope_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_12.stride.c+(0) * q_nope_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_14 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_12.stride, .addr = wukv_full_tile_12.addr + (((0) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_13 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_12.stride, .addr = q_wukv_acc_12.addr + (((0) / 64) * q_wukv_acc_12.stride.c+(0) * q_wukv_acc_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_12.stride.c+(0) * q_wukv_acc_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_13.addr, q_nope_head_14.addr, wukv_full_tile_14.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_14 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_12.stride, .addr = q_wukv_acc_12.addr + ((0) * q_wukv_acc_12.stride.c+(0) * q_wukv_acc_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_12.stride.c+(0) * q_wukv_acc_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_12.default_stride};
    __ppl_tensor_info q_wukv_head_13 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_12.stride, .addr = q_wukv_head_12.addr + ((0) * q_wukv_head_12.stride.c+(0) * q_wukv_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_12.stride.c+(0) * q_wukv_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_12.default_stride};
    tpu_bdc_cast(q_wukv_head_13.addr, q_wukv_acc_14.addr, &q_wukv_head_13.shape, (q_wukv_head_13.default_stride ? NULL : &q_wukv_head_13.stride), (q_wukv_acc_14.default_stride ? NULL : &q_wukv_acc_14.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_14 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_12.stride, .addr = q_wukv_head_12.addr + (((0) / 64) * q_wukv_head_12.stride.c+(0) * q_wukv_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_12.stride.c+(0) * q_wukv_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_10 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_8.stride, .addr = kv_cache_full_8.addr + (((0) / 64) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_7 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_6.stride, .addr = score_latent_h0_6.addr + (((0) / 64) * score_latent_h0_6.stride.c+(0) * score_latent_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_6.stride.c+(0) * score_latent_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_7.addr, q_wukv_head_14.addr, kv_cache_full_10.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_5 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_4.stride, .addr = q_rope_h0_4.addr + (((0) / 64) * q_rope_h0_4.stride.c+(0) * q_rope_h0_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_4.stride.c+(0) * q_rope_h0_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_10 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_8.stride, .addr = pe_score_tile_8.addr + (((0) / 64) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_7 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_6.stride, .addr = score_rope_h0_6.addr + (((0) / 64) * score_rope_h0_6.stride.c+(0) * score_rope_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_6.stride.c+(0) * score_rope_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_7.addr, q_rope_h0_5.addr, pe_score_tile_10.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_8.addr, score_latent_h0_6.addr, score_rope_h0_6.addr, &score_tile_8.shape, (score_tile_8.default_stride ? NULL : &score_tile_8.stride), (score_latent_h0_6.default_stride ? NULL : &score_latent_h0_6.stride), (score_rope_h0_6.default_stride ? NULL : &score_rope_h0_6.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_8.addr, score_tile_8.addr, mask_row_4.addr, &score_tile_8.shape, (score_tile_8.default_stride ? NULL : &score_tile_8.stride), (score_tile_8.default_stride ? NULL : &score_tile_8.stride), (mask_row_4.default_stride ? NULL : &mask_row_4.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_8.addr, score_tile_8.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_8.shape, (score_tile_8.default_stride ? NULL : &score_tile_8.stride), (score_tile_8.default_stride ? NULL : &score_tile_8.stride), DT_FP32);
    {
    scalar_t scores_max_4_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_4.addr, scores_max_4_scalar_f32, &scores_max_4.shape, (scores_max_4.default_stride ? NULL : &scores_max_4.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_4 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_4.stride, 0, &tmp_buffer_max_4.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_8.shape.n, score_tile_8.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_8.shape.n, score_tile_8.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_8.shape.n, score_tile_8.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_8.shape.n, score_tile_8.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_8.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, score_tile_8.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_8.shape.w) {
          dim4 fill_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, align_w - score_tile_8.shape.w};
          int elem_size = 4;
          int offset = score_tile_8.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_8.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_8.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_4_stride;
    tpu_aligned_stride(&scores_max_4_stride, 0, &scores_max_4.shape, DT_FP32);
    scores_max_4_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_8.addr, score_tile_8.addr, scores_max_4.addr, &score_tile_8.shape, (score_tile_8.default_stride ? NULL : &score_tile_8.stride), (score_tile_8.default_stride ? NULL : &score_tile_8.stride), &scores_max_4_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_4.addr);
    tpu_bdc_load_fp32_exp_table(table_4.addr);
    tpu_bdc_fp32_exp(score_tile_8.addr, score_tile_8.addr, work_score0_4.addr, work_score1_4.addr, coeff_4.addr, table_4.addr, &score_tile_8.shape);
    __ppl_tensor_info tmp_buffer_sum_4 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_4.stride, 0, &tmp_buffer_sum_4.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_8.shape.n, score_tile_8.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_8.shape.n, score_tile_8.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_8.shape.n, score_tile_8.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_8.shape.n, score_tile_8.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_8.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, score_tile_8.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_8.shape.w) {
          dim4 fill_shape = {score_tile_8.shape.n, score_tile_8.shape.c, 1, align_w - score_tile_8.shape.w};
          int elem_size = 4;
          int offset = score_tile_8.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_8.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_8.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_4.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_9 = {.shape = {1, 1, 1, 192} , .stride = score_tile_8.stride, .addr = score_tile_8.addr + ((0) * score_tile_8.stride.c+(0) * score_tile_8.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_8.stride.c+(0) * score_tile_8.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_8.default_stride};
    __ppl_tensor_info prob_tile_9 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_8.stride, .addr = prob_tile_8.addr + ((0) * prob_tile_8.stride.c+(0) * prob_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_8.stride.c+(0) * prob_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_8.default_stride};
    tpu_bdc_cast(prob_tile_9.addr, score_tile_9.addr, &prob_tile_9.shape, (prob_tile_9.default_stride ? NULL : &prob_tile_9.stride), (score_tile_9.default_stride ? NULL : &score_tile_9.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_8_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_8.addr, ctx_acc_8_scalar_f32, &ctx_acc_8.shape, (ctx_acc_8.default_stride ? NULL : &ctx_acc_8.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_8.addr, prob_tile_8.addr, kv_cache_full_8.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_4_stride;
    tpu_aligned_stride(&scores_sum_4_stride, 0, &scores_sum_4.shape, DT_FP32);
    scores_sum_4_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_8.addr, ctx_acc_8.addr, scores_sum_4.addr, &ctx_acc_8.shape, (ctx_acc_8.default_stride ? NULL : &ctx_acc_8.stride), (ctx_acc_8.default_stride ? NULL : &ctx_acc_8.stride), &scores_sum_4_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_9 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_8.stride, .addr = ctx_acc_8.addr + ((0) * ctx_acc_8.stride.c+(0) * ctx_acc_8.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_8.stride.c+(0) * ctx_acc_8.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_8.default_stride};
    __ppl_tensor_info ctx_head_13 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_12.stride, .addr = ctx_head_12.addr + ((0) * ctx_head_12.stride.c+(0) * ctx_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_12.stride.c+(0) * ctx_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_12.default_stride};
    tpu_bdc_cast(ctx_head_13.addr, ctx_acc_9.addr, &ctx_head_13.shape, (ctx_head_13.default_stride ? NULL : &ctx_head_13.stride), (ctx_acc_9.default_stride ? NULL : &ctx_acc_9.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_12.addr, out_part_12_scalar_f32, &out_part_12.shape, (out_part_12.default_stride ? NULL : &out_part_12.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_14 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_12.stride, .addr = ctx_head_12.addr + (((0) / 64) * ctx_head_12.stride.c+(0) * ctx_head_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_12.stride.c+(0) * ctx_head_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_15 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_12.stride, .addr = wukv_full_tile_12.addr + (((128) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_13 = {.shape = {1, 1, 1, 128}, .stride = out_part_12.stride, .addr = out_part_12.addr + (((0) / 64) * out_part_12.stride.c+(0) * out_part_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_12.stride.c+(0) * out_part_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_13.addr, ctx_head_14.addr, wukv_full_tile_15.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_14 = {.shape = {1, 1, 1, 128} , .stride = out_part_12.stride, .addr = out_part_12.addr + ((0) * out_part_12.stride.c+(0) * out_part_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_12.stride.c+(0) * out_part_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_12.default_stride};
    __ppl_tensor_info out_cast_13 = {.shape = {1, 1, 1, 128} , .stride = out_cast_12.stride, .addr = out_cast_12.addr + ((0) * out_cast_12.stride.c+(0) * out_cast_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_12.stride.c+(0) * out_cast_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_12.default_stride};
    tpu_bdc_cast(out_cast_13.addr, out_part_14.addr, &out_cast_13.shape, (out_cast_13.default_stride ? NULL : &out_cast_13.stride), (out_part_14.default_stride ? NULL : &out_part_14.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_14 = {.shape = {1, 1, 1, 128} , .stride = out_cast_12.stride, .addr = out_cast_12.addr + ((0) * out_cast_12.stride.c+(0) * out_cast_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_12.stride.c+(0) * out_cast_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_12.default_stride};
    __ppl_tensor_info OUT_4 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(512) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(512) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_4.addr, out_cast_14.addr, &OUT_4.shape, (OUT_4.default_stride ? NULL : &OUT_4.stride), (out_cast_14.default_stride ? NULL : &out_cast_14.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_15 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_15.stride, 0, &q_nope_head_15.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_15 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_15.stride, 0, &q_wukv_acc_15.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_15 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_15.stride, 0, &q_wukv_head_15.shape, DT_BFP16);
    __ppl_tensor_info score_tile_10 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_10.stride, 0, &score_tile_10.shape, DT_FP32);
    __ppl_tensor_info prob_tile_10 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_10.stride, 0, &prob_tile_10.shape, DT_BFP16);
    __ppl_tensor_info scores_max_5 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_5.stride, 0, &scores_max_5.shape, DT_FP32);
    __ppl_tensor_info scores_sum_5 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_5.stride, 0, &scores_sum_5.shape, DT_FP32);
    __ppl_tensor_info work_score0_5 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_5.stride, 0, &work_score0_5.shape, DT_FP32);
    __ppl_tensor_info work_score1_5 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_5.stride, 0, &work_score1_5.shape, DT_FP32);
    __ppl_tensor_info coeff_5 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_5.stride, 0, &coeff_5.shape, DT_FP32);
    __ppl_tensor_info table_5 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_5.stride, 0, &table_5.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_10 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_10.stride, 0, &ctx_acc_10.shape, DT_FP32);
    __ppl_tensor_info ctx_head_15 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_15.stride, 0, &ctx_head_15.shape, DT_BFP16);
    __ppl_tensor_info out_part_15 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_15.stride, 0, &out_part_15.shape, DT_FP32);
    __ppl_tensor_info out_cast_15 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_15.stride, 0, &out_cast_15.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_11 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_8.stride, .addr = q_upper_h1_8.addr + ((0) * q_upper_h1_8.stride.c+(0) * q_upper_h1_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_8.stride.c+(0) * q_upper_h1_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_8.default_stride};
    __ppl_tensor_info q_nope_head_16 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_15.stride, .addr = q_nope_head_15.addr + ((0) * q_nope_head_15.stride.c+(0) * q_nope_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_15.stride.c+(0) * q_nope_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_15.default_stride};
    tpu_bdc_cpy(q_nope_head_16.addr, q_upper_h1_11.addr, &q_nope_head_16.shape, (q_nope_head_16.default_stride ? NULL : &q_nope_head_16.stride), (q_upper_h1_11.default_stride ? NULL : &q_upper_h1_11.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_15.addr, q_wukv_acc_15_scalar_f32, &q_wukv_acc_15.shape, (q_wukv_acc_15.default_stride ? NULL : &q_wukv_acc_15.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_17 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_15.stride, .addr = q_nope_head_15.addr + (((0) / 64) * q_nope_head_15.stride.c+(0) * q_nope_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_15.stride.c+(0) * q_nope_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_16 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_12.stride, .addr = wukv_full_tile_12.addr + (((256) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_16 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_15.stride, .addr = q_wukv_acc_15.addr + (((0) / 64) * q_wukv_acc_15.stride.c+(0) * q_wukv_acc_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_15.stride.c+(0) * q_wukv_acc_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_16.addr, q_nope_head_17.addr, wukv_full_tile_16.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_17 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_15.stride, .addr = q_wukv_acc_15.addr + ((0) * q_wukv_acc_15.stride.c+(0) * q_wukv_acc_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_15.stride.c+(0) * q_wukv_acc_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_15.default_stride};
    __ppl_tensor_info q_wukv_head_16 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_15.stride, .addr = q_wukv_head_15.addr + ((0) * q_wukv_head_15.stride.c+(0) * q_wukv_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_15.stride.c+(0) * q_wukv_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_15.default_stride};
    tpu_bdc_cast(q_wukv_head_16.addr, q_wukv_acc_17.addr, &q_wukv_head_16.shape, (q_wukv_head_16.default_stride ? NULL : &q_wukv_head_16.stride), (q_wukv_acc_17.default_stride ? NULL : &q_wukv_acc_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_17 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_15.stride, .addr = q_wukv_head_15.addr + (((0) / 64) * q_wukv_head_15.stride.c+(0) * q_wukv_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_15.stride.c+(0) * q_wukv_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_11 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_8.stride, .addr = kv_cache_full_8.addr + (((0) / 64) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_8.stride.c+(0) * kv_cache_full_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_8 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_6.stride, .addr = score_latent_h0_6.addr + (((0) / 64) * score_latent_h0_6.stride.c+(0) * score_latent_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_6.stride.c+(0) * score_latent_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_8.addr, q_wukv_head_17.addr, kv_cache_full_11.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_5 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_4.stride, .addr = q_rope_h1_4.addr + (((0) / 64) * q_rope_h1_4.stride.c+(0) * q_rope_h1_4.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_4.stride.c+(0) * q_rope_h1_4.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_11 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_8.stride, .addr = pe_score_tile_8.addr + (((0) / 64) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_8.stride.c+(0) * pe_score_tile_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_8 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_6.stride, .addr = score_rope_h0_6.addr + (((0) / 64) * score_rope_h0_6.stride.c+(0) * score_rope_h0_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_6.stride.c+(0) * score_rope_h0_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_8.addr, q_rope_h1_5.addr, pe_score_tile_11.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_10.addr, score_latent_h0_6.addr, score_rope_h0_6.addr, &score_tile_10.shape, (score_tile_10.default_stride ? NULL : &score_tile_10.stride), (score_latent_h0_6.default_stride ? NULL : &score_latent_h0_6.stride), (score_rope_h0_6.default_stride ? NULL : &score_rope_h0_6.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_10.addr, score_tile_10.addr, mask_row_4.addr, &score_tile_10.shape, (score_tile_10.default_stride ? NULL : &score_tile_10.stride), (score_tile_10.default_stride ? NULL : &score_tile_10.stride), (mask_row_4.default_stride ? NULL : &mask_row_4.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_10.addr, score_tile_10.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_10.shape, (score_tile_10.default_stride ? NULL : &score_tile_10.stride), (score_tile_10.default_stride ? NULL : &score_tile_10.stride), DT_FP32);
    {
    scalar_t scores_max_5_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_5.addr, scores_max_5_scalar_f32, &scores_max_5.shape, (scores_max_5.default_stride ? NULL : &scores_max_5.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_5 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_5.stride, 0, &tmp_buffer_max_5.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_10.shape.n, score_tile_10.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_10.shape.n, score_tile_10.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_10.shape.n, score_tile_10.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_10.shape.n, score_tile_10.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_10.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, score_tile_10.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_10.shape.w) {
          dim4 fill_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, align_w - score_tile_10.shape.w};
          int elem_size = 4;
          int offset = score_tile_10.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_10.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_10.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_5_stride;
    tpu_aligned_stride(&scores_max_5_stride, 0, &scores_max_5.shape, DT_FP32);
    scores_max_5_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_10.addr, score_tile_10.addr, scores_max_5.addr, &score_tile_10.shape, (score_tile_10.default_stride ? NULL : &score_tile_10.stride), (score_tile_10.default_stride ? NULL : &score_tile_10.stride), &scores_max_5_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_5.addr);
    tpu_bdc_load_fp32_exp_table(table_5.addr);
    tpu_bdc_fp32_exp(score_tile_10.addr, score_tile_10.addr, work_score0_5.addr, work_score1_5.addr, coeff_5.addr, table_5.addr, &score_tile_10.shape);
    __ppl_tensor_info tmp_buffer_sum_5 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_5.stride, 0, &tmp_buffer_sum_5.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_10.shape.n, score_tile_10.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_10.shape.n, score_tile_10.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_10.shape.n, score_tile_10.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_10.shape.n, score_tile_10.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_10.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, score_tile_10.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_10.shape.w) {
          dim4 fill_shape = {score_tile_10.shape.n, score_tile_10.shape.c, 1, align_w - score_tile_10.shape.w};
          int elem_size = 4;
          int offset = score_tile_10.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_10.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_10.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_5.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_11 = {.shape = {1, 1, 1, 192} , .stride = score_tile_10.stride, .addr = score_tile_10.addr + ((0) * score_tile_10.stride.c+(0) * score_tile_10.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_10.stride.c+(0) * score_tile_10.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_10.default_stride};
    __ppl_tensor_info prob_tile_11 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_10.stride, .addr = prob_tile_10.addr + ((0) * prob_tile_10.stride.c+(0) * prob_tile_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_10.stride.c+(0) * prob_tile_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_10.default_stride};
    tpu_bdc_cast(prob_tile_11.addr, score_tile_11.addr, &prob_tile_11.shape, (prob_tile_11.default_stride ? NULL : &prob_tile_11.stride), (score_tile_11.default_stride ? NULL : &score_tile_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_10_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_10.addr, ctx_acc_10_scalar_f32, &ctx_acc_10.shape, (ctx_acc_10.default_stride ? NULL : &ctx_acc_10.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_10.addr, prob_tile_10.addr, kv_cache_full_8.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_5_stride;
    tpu_aligned_stride(&scores_sum_5_stride, 0, &scores_sum_5.shape, DT_FP32);
    scores_sum_5_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_10.addr, ctx_acc_10.addr, scores_sum_5.addr, &ctx_acc_10.shape, (ctx_acc_10.default_stride ? NULL : &ctx_acc_10.stride), (ctx_acc_10.default_stride ? NULL : &ctx_acc_10.stride), &scores_sum_5_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_11 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_10.stride, .addr = ctx_acc_10.addr + ((0) * ctx_acc_10.stride.c+(0) * ctx_acc_10.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_10.stride.c+(0) * ctx_acc_10.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_10.default_stride};
    __ppl_tensor_info ctx_head_16 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_15.stride, .addr = ctx_head_15.addr + ((0) * ctx_head_15.stride.c+(0) * ctx_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_15.stride.c+(0) * ctx_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_15.default_stride};
    tpu_bdc_cast(ctx_head_16.addr, ctx_acc_11.addr, &ctx_head_16.shape, (ctx_head_16.default_stride ? NULL : &ctx_head_16.stride), (ctx_acc_11.default_stride ? NULL : &ctx_acc_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_15.addr, out_part_15_scalar_f32, &out_part_15.shape, (out_part_15.default_stride ? NULL : &out_part_15.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_17 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_15.stride, .addr = ctx_head_15.addr + (((0) / 64) * ctx_head_15.stride.c+(0) * ctx_head_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_15.stride.c+(0) * ctx_head_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_17 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_12.stride, .addr = wukv_full_tile_12.addr + (((384) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_12.stride.c+(0) * wukv_full_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_16 = {.shape = {1, 1, 1, 128}, .stride = out_part_15.stride, .addr = out_part_15.addr + (((0) / 64) * out_part_15.stride.c+(0) * out_part_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_15.stride.c+(0) * out_part_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_16.addr, ctx_head_17.addr, wukv_full_tile_17.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_17 = {.shape = {1, 1, 1, 128} , .stride = out_part_15.stride, .addr = out_part_15.addr + ((0) * out_part_15.stride.c+(0) * out_part_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_15.stride.c+(0) * out_part_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_15.default_stride};
    __ppl_tensor_info out_cast_16 = {.shape = {1, 1, 1, 128} , .stride = out_cast_15.stride, .addr = out_cast_15.addr + ((0) * out_cast_15.stride.c+(0) * out_cast_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_15.stride.c+(0) * out_cast_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_15.default_stride};
    tpu_bdc_cast(out_cast_16.addr, out_part_17.addr, &out_cast_16.shape, (out_cast_16.default_stride ? NULL : &out_cast_16.stride), (out_part_17.default_stride ? NULL : &out_part_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_17 = {.shape = {1, 1, 1, 128} , .stride = out_cast_15.stride, .addr = out_cast_15.addr + ((0) * out_cast_15.stride.c+(0) * out_cast_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_15.stride.c+(0) * out_cast_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_15.default_stride};
    __ppl_tensor_info OUT_5 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(640) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(640) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_5.addr, out_cast_17.addr, &OUT_5.shape, (OUT_5.default_stride ? NULL : &OUT_5.stride), (out_cast_17.default_stride ? NULL : &out_cast_17.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_9.stride, 0, &q_upper_acc_h0_9.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_12.stride, 0, &q_upper_h0_12.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_9.stride, 0, &q_upper_acc_h1_9.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_12.stride, 0, &q_upper_h1_12.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_6.stride, 0, &q_rope_src_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_6.stride, 0, &q_rope_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_3.stride, 0, &q_rope_tmp0_h0_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_3.stride, 0, &q_rope_tmp1_h0_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_3.stride, 0, &q_rope_tmp2_h0_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_3.stride, 0, &q_rope_tmp3_h0_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_6.stride, 0, &q_rope_src_h1_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_6.stride, 0, &q_rope_h1_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_3.stride, 0, &q_rope_tmp0_h1_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_3.stride, 0, &q_rope_tmp1_h1_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_3.stride, 0, &q_rope_tmp2_h1_3.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_3.stride, 0, &q_rope_tmp3_h1_3.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_6.stride, 0, &cos_shared_6.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_6.stride, 0, &sin_shared_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_6.stride, 0, &pe_cur_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_3.stride, 0, &pe_cur_cos_3.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_3.stride, 0, &pe_cur_sin_3.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_3.stride, 0, &pe_cur_neg_3.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_3 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_3.stride, 0, &pe_cur_neg_sin_3.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_6.stride, 0, &pe_cur_rope_6.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_18 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_18.stride, 0, &wukv_full_tile_18.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_12 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_12.stride, 0, &kv_cache_full_12.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_12 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_12.stride, 0, &pe_score_tile_12.shape, DT_BFP16);
  __ppl_tensor_info mask_row_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_6.stride, 0, &mask_row_6.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_9.stride, 0, &score_latent_h0_9.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_9.stride, 0, &score_rope_h0_9.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_3.stride, 0, &score_latent_h1_3.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_3 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_3.stride, 0, &score_rope_h1_3.shape, DT_FP32);
  if (tpu_workitem_index() == 3) {
    __ppl_tensor_info COS_3 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_7 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_6.stride, .addr = cos_shared_6.addr + ((0) * cos_shared_6.stride.c+(0) * cos_shared_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_6.stride.c+(0) * cos_shared_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_6.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_7.addr, COS_3.addr, &cos_shared_7.shape, (cos_shared_7.default_stride ? NULL : &cos_shared_7.stride), (COS_3.default_stride ? NULL : &COS_3.stride), DT_BFP16);
    __ppl_tensor_info SIN_3 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_7 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_6.stride, .addr = sin_shared_6.addr + ((0) * sin_shared_6.stride.c+(0) * sin_shared_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_6.stride.c+(0) * sin_shared_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_6.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_7.addr, SIN_3.addr, &sin_shared_7.shape, (sin_shared_7.default_stride ? NULL : &sin_shared_7.stride), (SIN_3.default_stride ? NULL : &SIN_3.stride), DT_BFP16);
    for (int d_6 = 0; d_6 < 4; ++d_6) {
      __ppl_tensor_info kv_cur_tile_9 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_9.stride, 0, &kv_cur_tile_9.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_3 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_6 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_6 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_10 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_9.stride, .addr = kv_cur_tile_9.addr + ((0) * kv_cur_tile_9.stride.c+(0) * kv_cur_tile_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_9.stride.c+(0) * kv_cur_tile_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_9.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_10.addr, KV_CUR_3.addr, &kv_cur_tile_10.shape, (kv_cur_tile_10.default_stride ? NULL : &kv_cur_tile_10.stride), (KV_CUR_3.default_stride ? NULL : &KV_CUR_3.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_11 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_9.stride, .addr = kv_cur_tile_9.addr + ((0) * kv_cur_tile_9.stride.c+(0) * kv_cur_tile_9.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_9.stride.c+(0) * kv_cur_tile_9.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_9.default_stride};
      __ppl_tensor_info KVcache_OUT_6 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_6 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_6 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_6.addr, kv_cur_tile_11.addr, &KVcache_OUT_6.shape, (KVcache_OUT_6.default_stride ? NULL : &KVcache_OUT_6.stride), (kv_cur_tile_11.default_stride ? NULL : &kv_cur_tile_11.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_3 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_7 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_6.stride, .addr = pe_cur_6.addr + ((0) * pe_cur_6.stride.c+(0) * pe_cur_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_6.stride.c+(0) * pe_cur_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_6.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_7.addr, PE_CUR_3.addr, &pe_cur_7.shape, (pe_cur_7.default_stride ? NULL : &pe_cur_7.stride), (PE_CUR_3.default_stride ? NULL : &PE_CUR_3.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_3.addr, pe_cur_6.addr, cos_shared_6.addr, &pe_cur_cos_3.shape, (pe_cur_cos_3.default_stride ? NULL : &pe_cur_cos_3.stride), (pe_cur_6.default_stride ? NULL : &pe_cur_6.stride), (cos_shared_6.default_stride ? NULL : &cos_shared_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_3.addr, pe_cur_6.addr, sin_shared_6.addr, &pe_cur_sin_3.shape, (pe_cur_sin_3.default_stride ? NULL : &pe_cur_sin_3.stride), (pe_cur_6.default_stride ? NULL : &pe_cur_6.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_3_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_3_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_3_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_3.addr, pe_cur_6.addr, pe_cur_neg_3_scalar_DT_BFP16, &pe_cur_neg_3.shape, (pe_cur_neg_3.default_stride ? NULL : &pe_cur_neg_3.stride), (pe_cur_6.default_stride ? NULL : &pe_cur_6.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_3.addr, pe_cur_neg_3.addr, sin_shared_6.addr, &pe_cur_neg_sin_3.shape, (pe_cur_neg_sin_3.default_stride ? NULL : &pe_cur_neg_sin_3.stride), (pe_cur_neg_3.default_stride ? NULL : &pe_cur_neg_3.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_6.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_6.shape.n, .c = pe_cur_rope_6.shape.c, .h = pe_cur_rope_6.shape.h, .w = pe_cur_rope_6.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_6.addr, pe_cur_cos_3.addr, pe_cur_neg_sin_3.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_6.addr + 2, pe_cur_cos_3.addr + 2, pe_cur_sin_3.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_7 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_6.stride, .addr = pe_cur_rope_6.addr + ((0) * pe_cur_rope_6.stride.c+(0) * pe_cur_rope_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_6.stride.c+(0) * pe_cur_rope_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_6.default_stride};
    __ppl_tensor_info PEcache_OUT_6 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_6.addr, pe_cur_rope_7.addr, &PEcache_OUT_6.shape, (PEcache_OUT_6.default_stride ? NULL : &PEcache_OUT_6.stride), (pe_cur_rope_7.default_stride ? NULL : &pe_cur_rope_7.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_9.addr, q_upper_acc_h0_9_scalar_f32, &q_upper_acc_h0_9.shape, (q_upper_acc_h0_9.default_stride ? NULL : &q_upper_acc_h0_9.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_9.addr, q_upper_acc_h1_9_scalar_f32, &q_upper_acc_h1_9.shape, (q_upper_acc_h1_9.default_stride ? NULL : &q_upper_acc_h1_9.stride), DT_FP32);
    }
    for (int k_3 = 0; k_3 < 12; ++k_3) {
      __ppl_tensor_info q_tile_12 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_12.stride, 0, &q_tile_12.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_9 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_9.stride, 0, &wuq_fp8_9.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_24 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_24.stride, 0, &wuq_scale_stage_24.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_12 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_12.stride, 0, &wuq_tile_12.shape, DT_BFP16);
      __ppl_tensor_info Q_3 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_3 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_3 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_13 = {.shape = {1, 1, 1, 128} , .stride = q_tile_12.stride, .addr = q_tile_12.addr + ((0) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_12.default_stride};
      tpu_gdma_cpy_S2L(q_tile_13.addr, Q_3.addr, &q_tile_13.shape, (q_tile_13.default_stride ? NULL : &q_tile_13.stride), (Q_3.default_stride ? NULL : &Q_3.stride), DT_BFP16);
      __ppl_tensor_info WUQ_3 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((1152) * 1536+((k_3 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((1152) * 1536+((k_3 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_10 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_9.stride, .addr = wuq_fp8_9.addr + ((0) * wuq_fp8_9.stride.c+(0) * wuq_fp8_9.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_9.stride.c+(0) * wuq_fp8_9.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_9.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_10.addr, WUQ_3.addr, &wuq_fp8_10.shape, (wuq_fp8_10.default_stride ? NULL : &wuq_fp8_10.stride), (WUQ_3.default_stride ? NULL : &WUQ_3.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_54 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_54.stride, 0, &wuq_scale_scalar_54.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_18 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((9) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((9) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_55 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_54.stride, .addr = wuq_scale_scalar_54.addr + ((0) * wuq_scale_scalar_54.stride.c+(0) * wuq_scale_scalar_54.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_54.stride.c+(0) * wuq_scale_scalar_54.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_54.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_55.addr, WUQ_SCALE_18.addr, &wuq_scale_scalar_55.shape, (wuq_scale_scalar_55.default_stride ? NULL : &wuq_scale_scalar_55.stride), (WUQ_SCALE_18.default_stride ? NULL : &WUQ_SCALE_18.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_56 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_54.stride, .addr = wuq_scale_scalar_54.addr + (((0) / 64) * wuq_scale_scalar_54.stride.c+(0) * wuq_scale_scalar_54.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_54.stride.c+(0) * wuq_scale_scalar_54.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_25 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((0) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_25.addr, wuq_scale_scalar_56.addr, &wuq_scale_stage_25.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_57 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_57.stride, 0, &wuq_scale_scalar_57.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_19 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((9) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((9) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_58 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_57.stride, .addr = wuq_scale_scalar_57.addr + ((0) * wuq_scale_scalar_57.stride.c+(0) * wuq_scale_scalar_57.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_57.stride.c+(0) * wuq_scale_scalar_57.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_57.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_58.addr, WUQ_SCALE_19.addr, &wuq_scale_scalar_58.shape, (wuq_scale_scalar_58.default_stride ? NULL : &wuq_scale_scalar_58.stride), (WUQ_SCALE_19.default_stride ? NULL : &WUQ_SCALE_19.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_59 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_57.stride, .addr = wuq_scale_scalar_57.addr + (((0) / 64) * wuq_scale_scalar_57.stride.c+(0) * wuq_scale_scalar_57.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_57.stride.c+(0) * wuq_scale_scalar_57.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_26 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((64) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_26.addr, wuq_scale_scalar_59.addr, &wuq_scale_stage_26.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_60 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_60.stride, 0, &wuq_scale_scalar_60.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_20 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((10) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((10) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_61 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_60.stride, .addr = wuq_scale_scalar_60.addr + ((0) * wuq_scale_scalar_60.stride.c+(0) * wuq_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_60.stride.c+(0) * wuq_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_60.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_61.addr, WUQ_SCALE_20.addr, &wuq_scale_scalar_61.shape, (wuq_scale_scalar_61.default_stride ? NULL : &wuq_scale_scalar_61.stride), (WUQ_SCALE_20.default_stride ? NULL : &WUQ_SCALE_20.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_62 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_60.stride, .addr = wuq_scale_scalar_60.addr + (((0) / 64) * wuq_scale_scalar_60.stride.c+(0) * wuq_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_60.stride.c+(0) * wuq_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_27 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((128) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_27.addr, wuq_scale_scalar_62.addr, &wuq_scale_stage_27.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_63 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_63.stride, 0, &wuq_scale_scalar_63.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_21 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((10) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((10) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_64 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_63.stride, .addr = wuq_scale_scalar_63.addr + ((0) * wuq_scale_scalar_63.stride.c+(0) * wuq_scale_scalar_63.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_63.stride.c+(0) * wuq_scale_scalar_63.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_63.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_64.addr, WUQ_SCALE_21.addr, &wuq_scale_scalar_64.shape, (wuq_scale_scalar_64.default_stride ? NULL : &wuq_scale_scalar_64.stride), (WUQ_SCALE_21.default_stride ? NULL : &WUQ_SCALE_21.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_65 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_63.stride, .addr = wuq_scale_scalar_63.addr + (((0) / 64) * wuq_scale_scalar_63.stride.c+(0) * wuq_scale_scalar_63.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_63.stride.c+(0) * wuq_scale_scalar_63.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_28 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((192) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_28.addr, wuq_scale_scalar_65.addr, &wuq_scale_stage_28.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_66 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_66.stride, 0, &wuq_scale_scalar_66.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_22 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((11) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((11) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_67 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_66.stride, .addr = wuq_scale_scalar_66.addr + ((0) * wuq_scale_scalar_66.stride.c+(0) * wuq_scale_scalar_66.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_66.stride.c+(0) * wuq_scale_scalar_66.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_66.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_67.addr, WUQ_SCALE_22.addr, &wuq_scale_scalar_67.shape, (wuq_scale_scalar_67.default_stride ? NULL : &wuq_scale_scalar_67.stride), (WUQ_SCALE_22.default_stride ? NULL : &WUQ_SCALE_22.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_68 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_66.stride, .addr = wuq_scale_scalar_66.addr + (((0) / 64) * wuq_scale_scalar_66.stride.c+(0) * wuq_scale_scalar_66.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_66.stride.c+(0) * wuq_scale_scalar_66.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_29 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((256) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_29.addr, wuq_scale_scalar_68.addr, &wuq_scale_stage_29.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_69 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_69.stride, 0, &wuq_scale_scalar_69.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_23 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((11) * 12+(k_3) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((11) * 12+(k_3) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_70 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_69.stride, .addr = wuq_scale_scalar_69.addr + ((0) * wuq_scale_scalar_69.stride.c+(0) * wuq_scale_scalar_69.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_69.stride.c+(0) * wuq_scale_scalar_69.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_69.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_70.addr, WUQ_SCALE_23.addr, &wuq_scale_scalar_70.shape, (wuq_scale_scalar_70.default_stride ? NULL : &wuq_scale_scalar_70.stride), (WUQ_SCALE_23.default_stride ? NULL : &WUQ_SCALE_23.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_71 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_69.stride, .addr = wuq_scale_scalar_69.addr + (((0) / 64) * wuq_scale_scalar_69.stride.c+(0) * wuq_scale_scalar_69.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_69.stride.c+(0) * wuq_scale_scalar_69.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_30 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((320) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_30.addr, wuq_scale_scalar_71.addr, &wuq_scale_stage_30.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_11 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_9.stride, .addr = wuq_fp8_9.addr + (((0) / 64) * wuq_fp8_9.stride.c+(0) * wuq_fp8_9.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_9.stride.c+(0) * wuq_fp8_9.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_31 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_24.stride, .addr = wuq_scale_stage_24.addr + (((0) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_24.stride.c+(0) * wuq_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_13 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_12.stride, .addr = wuq_tile_12.addr + (((0) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_13.addr, wuq_fp8_11.addr, &wuq_tile_13.shape, (wuq_tile_13.default_stride ? NULL : &wuq_tile_13.stride), (wuq_fp8_11.default_stride ? NULL : &wuq_fp8_11.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_31_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_31_block_stride, 0, &wuq_scale_stage_31.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_31_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_13.addr, wuq_tile_13.addr, wuq_scale_stage_31.addr, &wuq_tile_13.shape, (wuq_tile_13.default_stride ? NULL : &wuq_tile_13.stride), (wuq_tile_13.default_stride ? NULL : &wuq_tile_13.stride), &wuq_scale_stage_31_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_14 = {.shape = {1, 1, 1, 128}, .stride = q_tile_12.stride, .addr = q_tile_12.addr + (((0) / 64) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_14 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_12.stride, .addr = wuq_tile_12.addr + (((0) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_10 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_9.stride, .addr = q_upper_acc_h0_9.addr + (((0) / 64) * q_upper_acc_h0_9.stride.c+(0) * q_upper_acc_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_9.stride.c+(0) * q_upper_acc_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_10.addr, q_tile_14.addr, wuq_tile_14.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_15 = {.shape = {1, 1, 1, 128}, .stride = q_tile_12.stride, .addr = q_tile_12.addr + (((0) / 64) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_12.stride.c+(0) * q_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_15 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_12.stride, .addr = wuq_tile_12.addr + (((192) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_12.stride.c+(0) * wuq_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_10 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_9.stride, .addr = q_upper_acc_h1_9.addr + (((0) / 64) * q_upper_acc_h1_9.stride.c+(0) * q_upper_acc_h1_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_9.stride.c+(0) * q_upper_acc_h1_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_10.addr, q_tile_15.addr, wuq_tile_15.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_11 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_9.stride, .addr = q_upper_acc_h0_9.addr + ((0) * q_upper_acc_h0_9.stride.c+(0) * q_upper_acc_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_9.stride.c+(0) * q_upper_acc_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_9.default_stride};
    __ppl_tensor_info q_upper_h0_13 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_12.stride, .addr = q_upper_h0_12.addr + ((0) * q_upper_h0_12.stride.c+(0) * q_upper_h0_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_12.stride.c+(0) * q_upper_h0_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_12.default_stride};
    tpu_bdc_cast(q_upper_h0_13.addr, q_upper_acc_h0_11.addr, &q_upper_h0_13.shape, (q_upper_h0_13.default_stride ? NULL : &q_upper_h0_13.stride), (q_upper_acc_h0_11.default_stride ? NULL : &q_upper_acc_h0_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_11 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_9.stride, .addr = q_upper_acc_h1_9.addr + ((0) * q_upper_acc_h1_9.stride.c+(0) * q_upper_acc_h1_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_9.stride.c+(0) * q_upper_acc_h1_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_9.default_stride};
    __ppl_tensor_info q_upper_h1_13 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_12.stride, .addr = q_upper_h1_12.addr + ((0) * q_upper_h1_12.stride.c+(0) * q_upper_h1_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_12.stride.c+(0) * q_upper_h1_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_12.default_stride};
    tpu_bdc_cast(q_upper_h1_13.addr, q_upper_acc_h1_11.addr, &q_upper_h1_13.shape, (q_upper_h1_13.default_stride ? NULL : &q_upper_h1_13.stride), (q_upper_acc_h1_11.default_stride ? NULL : &q_upper_acc_h1_11.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_7 = 0; d_7 < 4; ++d_7) {
      __ppl_tensor_info wukv_fp8_9 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_9.stride, 0, &wukv_fp8_9.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_18 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_18.stride, 0, &wukv_scale_stage_18.shape, DT_BFP16);
      __ppl_tensor_info WUKV_3 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((1536) * 512+((d_7 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((1536) * 512+((d_7 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_10 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_9.stride, .addr = wukv_fp8_9.addr + ((0) * wukv_fp8_9.stride.c+(0) * wukv_fp8_9.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_9.stride.c+(0) * wukv_fp8_9.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_9.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_10.addr, WUKV_3.addr, &wukv_fp8_10.shape, (wukv_fp8_10.default_stride ? NULL : &wukv_fp8_10.stride), (WUKV_3.default_stride ? NULL : &WUKV_3.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_30 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_30.stride, 0, &wukv_scale_scalar_30.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_12 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((12) * 4+(d_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((12) * 4+(d_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_31 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_30.stride, .addr = wukv_scale_scalar_30.addr + ((0) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_30.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_31.addr, WUKV_SCALE_12.addr, &wukv_scale_scalar_31.shape, (wukv_scale_scalar_31.default_stride ? NULL : &wukv_scale_scalar_31.stride), (WUKV_SCALE_12.default_stride ? NULL : &WUKV_SCALE_12.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_32 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_30.stride, .addr = wukv_scale_scalar_30.addr + (((0) / 64) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_19 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_18.stride, .addr = wukv_scale_stage_18.addr + (((0) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_19_chunk = wukv_scale_stage_19;
      wukv_scale_stage_19_chunk.shape.c = 64;
      wukv_scale_stage_19_chunk.addr = wukv_scale_stage_19.addr + ((0) * wukv_scale_stage_19.stride.c) * 2;
      wukv_scale_stage_19_chunk.offset = wukv_scale_stage_19.offset + ((0) * wukv_scale_stage_19.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_19_chunk.addr, wukv_scale_scalar_32.addr, &wukv_scale_stage_19_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_19_chunk_1 = wukv_scale_stage_19;
      wukv_scale_stage_19_chunk_1.shape.c = 64;
      wukv_scale_stage_19_chunk_1.addr = wukv_scale_stage_19.addr + ((1) * wukv_scale_stage_19.stride.c) * 2;
      wukv_scale_stage_19_chunk_1.offset = wukv_scale_stage_19.offset + ((1) * wukv_scale_stage_19.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_19_chunk_1.addr, wukv_scale_scalar_32.addr, &wukv_scale_stage_19_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_13 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((13) * 4+(d_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((13) * 4+(d_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_33 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_30.stride, .addr = wukv_scale_scalar_30.addr + ((0) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_30.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_33.addr, WUKV_SCALE_13.addr, &wukv_scale_scalar_33.shape, (wukv_scale_scalar_33.default_stride ? NULL : &wukv_scale_scalar_33.stride), (WUKV_SCALE_13.default_stride ? NULL : &WUKV_SCALE_13.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_34 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_30.stride, .addr = wukv_scale_scalar_30.addr + (((0) / 64) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_30.stride.c+(0) * wukv_scale_scalar_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_20 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_18.stride, .addr = wukv_scale_stage_18.addr + (((128) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_20_chunk = wukv_scale_stage_20;
      wukv_scale_stage_20_chunk.shape.c = 64;
      wukv_scale_stage_20_chunk.addr = wukv_scale_stage_20.addr + ((0) * wukv_scale_stage_20.stride.c) * 2;
      wukv_scale_stage_20_chunk.offset = wukv_scale_stage_20.offset + ((0) * wukv_scale_stage_20.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_20_chunk.addr, wukv_scale_scalar_34.addr, &wukv_scale_stage_20_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_20_chunk_1 = wukv_scale_stage_20;
      wukv_scale_stage_20_chunk_1.shape.c = 64;
      wukv_scale_stage_20_chunk_1.addr = wukv_scale_stage_20.addr + ((1) * wukv_scale_stage_20.stride.c) * 2;
      wukv_scale_stage_20_chunk_1.offset = wukv_scale_stage_20.offset + ((1) * wukv_scale_stage_20.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_20_chunk_1.addr, wukv_scale_scalar_34.addr, &wukv_scale_stage_20_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_35 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_35.stride, 0, &wukv_scale_scalar_35.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_14 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((14) * 4+(d_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((14) * 4+(d_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_36 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_35.stride, .addr = wukv_scale_scalar_35.addr + ((0) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_35.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_36.addr, WUKV_SCALE_14.addr, &wukv_scale_scalar_36.shape, (wukv_scale_scalar_36.default_stride ? NULL : &wukv_scale_scalar_36.stride), (WUKV_SCALE_14.default_stride ? NULL : &WUKV_SCALE_14.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_37 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_35.stride, .addr = wukv_scale_scalar_35.addr + (((0) / 64) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_21 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_18.stride, .addr = wukv_scale_stage_18.addr + (((256) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_21_chunk = wukv_scale_stage_21;
      wukv_scale_stage_21_chunk.shape.c = 64;
      wukv_scale_stage_21_chunk.addr = wukv_scale_stage_21.addr + ((0) * wukv_scale_stage_21.stride.c) * 2;
      wukv_scale_stage_21_chunk.offset = wukv_scale_stage_21.offset + ((0) * wukv_scale_stage_21.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_21_chunk.addr, wukv_scale_scalar_37.addr, &wukv_scale_stage_21_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_21_chunk_1 = wukv_scale_stage_21;
      wukv_scale_stage_21_chunk_1.shape.c = 64;
      wukv_scale_stage_21_chunk_1.addr = wukv_scale_stage_21.addr + ((1) * wukv_scale_stage_21.stride.c) * 2;
      wukv_scale_stage_21_chunk_1.offset = wukv_scale_stage_21.offset + ((1) * wukv_scale_stage_21.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_21_chunk_1.addr, wukv_scale_scalar_37.addr, &wukv_scale_stage_21_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_15 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((15) * 4+(d_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((15) * 4+(d_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_38 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_35.stride, .addr = wukv_scale_scalar_35.addr + ((0) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_35.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_38.addr, WUKV_SCALE_15.addr, &wukv_scale_scalar_38.shape, (wukv_scale_scalar_38.default_stride ? NULL : &wukv_scale_scalar_38.stride), (WUKV_SCALE_15.default_stride ? NULL : &WUKV_SCALE_15.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_39 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_35.stride, .addr = wukv_scale_scalar_35.addr + (((0) / 64) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_35.stride.c+(0) * wukv_scale_scalar_35.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_22 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_18.stride, .addr = wukv_scale_stage_18.addr + (((384) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_22_chunk = wukv_scale_stage_22;
      wukv_scale_stage_22_chunk.shape.c = 64;
      wukv_scale_stage_22_chunk.addr = wukv_scale_stage_22.addr + ((0) * wukv_scale_stage_22.stride.c) * 2;
      wukv_scale_stage_22_chunk.offset = wukv_scale_stage_22.offset + ((0) * wukv_scale_stage_22.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_22_chunk.addr, wukv_scale_scalar_39.addr, &wukv_scale_stage_22_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_22_chunk_1 = wukv_scale_stage_22;
      wukv_scale_stage_22_chunk_1.shape.c = 64;
      wukv_scale_stage_22_chunk_1.addr = wukv_scale_stage_22.addr + ((1) * wukv_scale_stage_22.stride.c) * 2;
      wukv_scale_stage_22_chunk_1.offset = wukv_scale_stage_22.offset + ((1) * wukv_scale_stage_22.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_22_chunk_1.addr, wukv_scale_scalar_39.addr, &wukv_scale_stage_22_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_11 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_9.stride, .addr = wukv_fp8_9.addr + (((0) / 64) * wukv_fp8_9.stride.c+(0) * wukv_fp8_9.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_9.stride.c+(0) * wukv_fp8_9.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_23 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_18.stride, .addr = wukv_scale_stage_18.addr + (((0) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_18.stride.c+(0) * wukv_scale_stage_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_19 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_18.stride, .addr = wukv_full_tile_18.addr + (((0) / 64) * wukv_full_tile_18.stride.c+((d_7 * 128)) * wukv_full_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_18.stride.c+((d_7 * 128)) * wukv_full_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_19.addr, wukv_fp8_11.addr, &wukv_full_tile_19.shape, (wukv_full_tile_19.default_stride ? NULL : &wukv_full_tile_19.stride), (wukv_fp8_11.default_stride ? NULL : &wukv_fp8_11.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_23_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_23_block_stride, 0, &wukv_scale_stage_23.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_23_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_19.addr, wukv_full_tile_19.addr, wukv_scale_stage_23.addr, &wukv_full_tile_19.shape, (wukv_full_tile_19.default_stride ? NULL : &wukv_full_tile_19.stride), (wukv_full_tile_19.default_stride ? NULL : &wukv_full_tile_19.stride), &wukv_scale_stage_23_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_7 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_13 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_12.stride, .addr = kv_cache_full_12.addr + ((0) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_12.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_13.addr, KVcache_OUT_7.addr, &kv_cache_full_13.shape, (kv_cache_full_13.default_stride ? NULL : &kv_cache_full_13.stride), (KVcache_OUT_7.default_stride ? NULL : &KVcache_OUT_7.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_7 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_13 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_12.stride, .addr = pe_score_tile_12.addr + ((0) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_12.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_13.addr, PEcache_OUT_7.addr, &pe_score_tile_13.shape, (pe_score_tile_13.default_stride ? NULL : &pe_score_tile_13.stride), (PEcache_OUT_7.default_stride ? NULL : &PEcache_OUT_7.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_3 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_7 = {.shape = {1, 1, 1, 192} , .stride = mask_row_6.stride, .addr = mask_row_6.addr + ((0) * mask_row_6.stride.c+(0) * mask_row_6.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_6.stride.c+(0) * mask_row_6.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_6.default_stride};
    tpu_gdma_cpy_S2L(mask_row_7.addr, DECODE_MASK_3.addr, &mask_row_7.shape, (mask_row_7.default_stride ? NULL : &mask_row_7.stride), (DECODE_MASK_3.default_stride ? NULL : &DECODE_MASK_3.stride), DT_FP32);
    {
    scalar_t score_latent_h0_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_9.addr, score_latent_h0_9_scalar_f32, &score_latent_h0_9.shape, (score_latent_h0_9.default_stride ? NULL : &score_latent_h0_9.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_9_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_9.addr, score_rope_h0_9_scalar_f32, &score_rope_h0_9.shape, (score_rope_h0_9.default_stride ? NULL : &score_rope_h0_9.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_3.addr, score_latent_h1_3_scalar_f32, &score_latent_h1_3.shape, (score_latent_h1_3.default_stride ? NULL : &score_latent_h1_3.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_3_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_3.addr, score_rope_h1_3_scalar_f32, &score_rope_h1_3.shape, (score_rope_h1_3.default_stride ? NULL : &score_rope_h1_3.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_14 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_12.stride, .addr = q_upper_h0_12.addr + ((0) * q_upper_h0_12.stride.c+(128) * q_upper_h0_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_12.stride.c+(128) * q_upper_h0_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_12.default_stride};
    __ppl_tensor_info q_rope_src_h0_7 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_6.stride, .addr = q_rope_src_h0_6.addr + ((0) * q_rope_src_h0_6.stride.c+(0) * q_rope_src_h0_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_6.stride.c+(0) * q_rope_src_h0_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_6.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_7.addr, q_upper_h0_14.addr, &q_rope_src_h0_7.shape, (q_rope_src_h0_7.default_stride ? NULL : &q_rope_src_h0_7.stride), (q_upper_h0_14.default_stride ? NULL : &q_upper_h0_14.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_3.addr, q_rope_src_h0_6.addr, cos_shared_6.addr, &q_rope_tmp0_h0_3.shape, (q_rope_tmp0_h0_3.default_stride ? NULL : &q_rope_tmp0_h0_3.stride), (q_rope_src_h0_6.default_stride ? NULL : &q_rope_src_h0_6.stride), (cos_shared_6.default_stride ? NULL : &cos_shared_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_3.addr, q_rope_src_h0_6.addr, sin_shared_6.addr, &q_rope_tmp1_h0_3.shape, (q_rope_tmp1_h0_3.default_stride ? NULL : &q_rope_tmp1_h0_3.stride), (q_rope_src_h0_6.default_stride ? NULL : &q_rope_src_h0_6.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_3_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_3_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_3_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_3.addr, q_rope_src_h0_6.addr, q_rope_tmp2_h0_3_scalar_DT_BFP16, &q_rope_tmp2_h0_3.shape, (q_rope_tmp2_h0_3.default_stride ? NULL : &q_rope_tmp2_h0_3.stride), (q_rope_src_h0_6.default_stride ? NULL : &q_rope_src_h0_6.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_3.addr, q_rope_tmp2_h0_3.addr, sin_shared_6.addr, &q_rope_tmp3_h0_3.shape, (q_rope_tmp3_h0_3.default_stride ? NULL : &q_rope_tmp3_h0_3.stride), (q_rope_tmp2_h0_3.default_stride ? NULL : &q_rope_tmp2_h0_3.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_6.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_6.shape.n, .c = q_rope_h0_6.shape.c, .h = q_rope_h0_6.shape.h, .w = q_rope_h0_6.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_6.addr, q_rope_tmp0_h0_3.addr, q_rope_tmp3_h0_3.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_6.addr + 2, q_rope_tmp0_h0_3.addr + 2, q_rope_tmp1_h0_3.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_14 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_12.stride, .addr = q_upper_h1_12.addr + ((0) * q_upper_h1_12.stride.c+(128) * q_upper_h1_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_12.stride.c+(128) * q_upper_h1_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_12.default_stride};
    __ppl_tensor_info q_rope_src_h1_7 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_6.stride, .addr = q_rope_src_h1_6.addr + ((0) * q_rope_src_h1_6.stride.c+(0) * q_rope_src_h1_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_6.stride.c+(0) * q_rope_src_h1_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_6.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_7.addr, q_upper_h1_14.addr, &q_rope_src_h1_7.shape, (q_rope_src_h1_7.default_stride ? NULL : &q_rope_src_h1_7.stride), (q_upper_h1_14.default_stride ? NULL : &q_upper_h1_14.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_3.addr, q_rope_src_h1_6.addr, cos_shared_6.addr, &q_rope_tmp0_h1_3.shape, (q_rope_tmp0_h1_3.default_stride ? NULL : &q_rope_tmp0_h1_3.stride), (q_rope_src_h1_6.default_stride ? NULL : &q_rope_src_h1_6.stride), (cos_shared_6.default_stride ? NULL : &cos_shared_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_3.addr, q_rope_src_h1_6.addr, sin_shared_6.addr, &q_rope_tmp1_h1_3.shape, (q_rope_tmp1_h1_3.default_stride ? NULL : &q_rope_tmp1_h1_3.stride), (q_rope_src_h1_6.default_stride ? NULL : &q_rope_src_h1_6.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_3_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_3_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_3_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_3.addr, q_rope_src_h1_6.addr, q_rope_tmp2_h1_3_scalar_DT_BFP16, &q_rope_tmp2_h1_3.shape, (q_rope_tmp2_h1_3.default_stride ? NULL : &q_rope_tmp2_h1_3.stride), (q_rope_src_h1_6.default_stride ? NULL : &q_rope_src_h1_6.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_3.addr, q_rope_tmp2_h1_3.addr, sin_shared_6.addr, &q_rope_tmp3_h1_3.shape, (q_rope_tmp3_h1_3.default_stride ? NULL : &q_rope_tmp3_h1_3.stride), (q_rope_tmp2_h1_3.default_stride ? NULL : &q_rope_tmp2_h1_3.stride), (sin_shared_6.default_stride ? NULL : &sin_shared_6.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_6.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_6.shape.n, .c = q_rope_h1_6.shape.c, .h = q_rope_h1_6.shape.h, .w = q_rope_h1_6.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_6.addr, q_rope_tmp0_h1_3.addr, q_rope_tmp3_h1_3.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_6.addr + 2, q_rope_tmp0_h1_3.addr + 2, q_rope_tmp1_h1_3.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_18 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_18.stride, 0, &q_nope_head_18.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_18 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_18.stride, 0, &q_wukv_acc_18.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_18 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_18.stride, 0, &q_wukv_head_18.shape, DT_BFP16);
    __ppl_tensor_info score_tile_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_12.stride, 0, &score_tile_12.shape, DT_FP32);
    __ppl_tensor_info prob_tile_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_12.stride, 0, &prob_tile_12.shape, DT_BFP16);
    __ppl_tensor_info scores_max_6 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_6.stride, 0, &scores_max_6.shape, DT_FP32);
    __ppl_tensor_info scores_sum_6 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_6.stride, 0, &scores_sum_6.shape, DT_FP32);
    __ppl_tensor_info work_score0_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_6.stride, 0, &work_score0_6.shape, DT_FP32);
    __ppl_tensor_info work_score1_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_6.stride, 0, &work_score1_6.shape, DT_FP32);
    __ppl_tensor_info coeff_6 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_6.stride, 0, &coeff_6.shape, DT_FP32);
    __ppl_tensor_info table_6 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_6.stride, 0, &table_6.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_12 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_12.stride, 0, &ctx_acc_12.shape, DT_FP32);
    __ppl_tensor_info ctx_head_18 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_18.stride, 0, &ctx_head_18.shape, DT_BFP16);
    __ppl_tensor_info out_part_18 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_18.stride, 0, &out_part_18.shape, DT_FP32);
    __ppl_tensor_info out_cast_18 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_18.stride, 0, &out_cast_18.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_15 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_12.stride, .addr = q_upper_h0_12.addr + ((0) * q_upper_h0_12.stride.c+(0) * q_upper_h0_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_12.stride.c+(0) * q_upper_h0_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_12.default_stride};
    __ppl_tensor_info q_nope_head_19 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_18.stride, .addr = q_nope_head_18.addr + ((0) * q_nope_head_18.stride.c+(0) * q_nope_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_18.stride.c+(0) * q_nope_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_18.default_stride};
    tpu_bdc_cpy(q_nope_head_19.addr, q_upper_h0_15.addr, &q_nope_head_19.shape, (q_nope_head_19.default_stride ? NULL : &q_nope_head_19.stride), (q_upper_h0_15.default_stride ? NULL : &q_upper_h0_15.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_18.addr, q_wukv_acc_18_scalar_f32, &q_wukv_acc_18.shape, (q_wukv_acc_18.default_stride ? NULL : &q_wukv_acc_18.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_20 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_18.stride, .addr = q_nope_head_18.addr + (((0) / 64) * q_nope_head_18.stride.c+(0) * q_nope_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_18.stride.c+(0) * q_nope_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_20 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_18.stride, .addr = wukv_full_tile_18.addr + (((0) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_19 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_18.stride, .addr = q_wukv_acc_18.addr + (((0) / 64) * q_wukv_acc_18.stride.c+(0) * q_wukv_acc_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_18.stride.c+(0) * q_wukv_acc_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_19.addr, q_nope_head_20.addr, wukv_full_tile_20.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_20 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_18.stride, .addr = q_wukv_acc_18.addr + ((0) * q_wukv_acc_18.stride.c+(0) * q_wukv_acc_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_18.stride.c+(0) * q_wukv_acc_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_18.default_stride};
    __ppl_tensor_info q_wukv_head_19 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_18.stride, .addr = q_wukv_head_18.addr + ((0) * q_wukv_head_18.stride.c+(0) * q_wukv_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_18.stride.c+(0) * q_wukv_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_18.default_stride};
    tpu_bdc_cast(q_wukv_head_19.addr, q_wukv_acc_20.addr, &q_wukv_head_19.shape, (q_wukv_head_19.default_stride ? NULL : &q_wukv_head_19.stride), (q_wukv_acc_20.default_stride ? NULL : &q_wukv_acc_20.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_20 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_18.stride, .addr = q_wukv_head_18.addr + (((0) / 64) * q_wukv_head_18.stride.c+(0) * q_wukv_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_18.stride.c+(0) * q_wukv_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_14 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_12.stride, .addr = kv_cache_full_12.addr + (((0) / 64) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_10 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_9.stride, .addr = score_latent_h0_9.addr + (((0) / 64) * score_latent_h0_9.stride.c+(0) * score_latent_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_9.stride.c+(0) * score_latent_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_10.addr, q_wukv_head_20.addr, kv_cache_full_14.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_7 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_6.stride, .addr = q_rope_h0_6.addr + (((0) / 64) * q_rope_h0_6.stride.c+(0) * q_rope_h0_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_6.stride.c+(0) * q_rope_h0_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_14 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_12.stride, .addr = pe_score_tile_12.addr + (((0) / 64) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_10 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_9.stride, .addr = score_rope_h0_9.addr + (((0) / 64) * score_rope_h0_9.stride.c+(0) * score_rope_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_9.stride.c+(0) * score_rope_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_10.addr, q_rope_h0_7.addr, pe_score_tile_14.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_12.addr, score_latent_h0_9.addr, score_rope_h0_9.addr, &score_tile_12.shape, (score_tile_12.default_stride ? NULL : &score_tile_12.stride), (score_latent_h0_9.default_stride ? NULL : &score_latent_h0_9.stride), (score_rope_h0_9.default_stride ? NULL : &score_rope_h0_9.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_12.addr, score_tile_12.addr, mask_row_6.addr, &score_tile_12.shape, (score_tile_12.default_stride ? NULL : &score_tile_12.stride), (score_tile_12.default_stride ? NULL : &score_tile_12.stride), (mask_row_6.default_stride ? NULL : &mask_row_6.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_12.addr, score_tile_12.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_12.shape, (score_tile_12.default_stride ? NULL : &score_tile_12.stride), (score_tile_12.default_stride ? NULL : &score_tile_12.stride), DT_FP32);
    {
    scalar_t scores_max_6_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_6.addr, scores_max_6_scalar_f32, &scores_max_6.shape, (scores_max_6.default_stride ? NULL : &scores_max_6.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_6 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_6.stride, 0, &tmp_buffer_max_6.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_12.shape.n, score_tile_12.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_12.shape.n, score_tile_12.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_12.shape.n, score_tile_12.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_12.shape.n, score_tile_12.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_12.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, score_tile_12.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_12.shape.w) {
          dim4 fill_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, align_w - score_tile_12.shape.w};
          int elem_size = 4;
          int offset = score_tile_12.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_12.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_12.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_6_stride;
    tpu_aligned_stride(&scores_max_6_stride, 0, &scores_max_6.shape, DT_FP32);
    scores_max_6_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_12.addr, score_tile_12.addr, scores_max_6.addr, &score_tile_12.shape, (score_tile_12.default_stride ? NULL : &score_tile_12.stride), (score_tile_12.default_stride ? NULL : &score_tile_12.stride), &scores_max_6_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_6.addr);
    tpu_bdc_load_fp32_exp_table(table_6.addr);
    tpu_bdc_fp32_exp(score_tile_12.addr, score_tile_12.addr, work_score0_6.addr, work_score1_6.addr, coeff_6.addr, table_6.addr, &score_tile_12.shape);
    __ppl_tensor_info tmp_buffer_sum_6 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_6.stride, 0, &tmp_buffer_sum_6.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_12.shape.n, score_tile_12.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_12.shape.n, score_tile_12.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_12.shape.n, score_tile_12.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_12.shape.n, score_tile_12.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_12.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, score_tile_12.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_12.shape.w) {
          dim4 fill_shape = {score_tile_12.shape.n, score_tile_12.shape.c, 1, align_w - score_tile_12.shape.w};
          int elem_size = 4;
          int offset = score_tile_12.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_12.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_12.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_6.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_13 = {.shape = {1, 1, 1, 192} , .stride = score_tile_12.stride, .addr = score_tile_12.addr + ((0) * score_tile_12.stride.c+(0) * score_tile_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_12.stride.c+(0) * score_tile_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_12.default_stride};
    __ppl_tensor_info prob_tile_13 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_12.stride, .addr = prob_tile_12.addr + ((0) * prob_tile_12.stride.c+(0) * prob_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_12.stride.c+(0) * prob_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_12.default_stride};
    tpu_bdc_cast(prob_tile_13.addr, score_tile_13.addr, &prob_tile_13.shape, (prob_tile_13.default_stride ? NULL : &prob_tile_13.stride), (score_tile_13.default_stride ? NULL : &score_tile_13.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_12.addr, ctx_acc_12_scalar_f32, &ctx_acc_12.shape, (ctx_acc_12.default_stride ? NULL : &ctx_acc_12.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_12.addr, prob_tile_12.addr, kv_cache_full_12.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_6_stride;
    tpu_aligned_stride(&scores_sum_6_stride, 0, &scores_sum_6.shape, DT_FP32);
    scores_sum_6_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_12.addr, ctx_acc_12.addr, scores_sum_6.addr, &ctx_acc_12.shape, (ctx_acc_12.default_stride ? NULL : &ctx_acc_12.stride), (ctx_acc_12.default_stride ? NULL : &ctx_acc_12.stride), &scores_sum_6_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_13 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_12.stride, .addr = ctx_acc_12.addr + ((0) * ctx_acc_12.stride.c+(0) * ctx_acc_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_12.stride.c+(0) * ctx_acc_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_12.default_stride};
    __ppl_tensor_info ctx_head_19 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_18.stride, .addr = ctx_head_18.addr + ((0) * ctx_head_18.stride.c+(0) * ctx_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_18.stride.c+(0) * ctx_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_18.default_stride};
    tpu_bdc_cast(ctx_head_19.addr, ctx_acc_13.addr, &ctx_head_19.shape, (ctx_head_19.default_stride ? NULL : &ctx_head_19.stride), (ctx_acc_13.default_stride ? NULL : &ctx_acc_13.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_18.addr, out_part_18_scalar_f32, &out_part_18.shape, (out_part_18.default_stride ? NULL : &out_part_18.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_20 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_18.stride, .addr = ctx_head_18.addr + (((0) / 64) * ctx_head_18.stride.c+(0) * ctx_head_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_18.stride.c+(0) * ctx_head_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_21 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_18.stride, .addr = wukv_full_tile_18.addr + (((128) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_19 = {.shape = {1, 1, 1, 128}, .stride = out_part_18.stride, .addr = out_part_18.addr + (((0) / 64) * out_part_18.stride.c+(0) * out_part_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_18.stride.c+(0) * out_part_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_19.addr, ctx_head_20.addr, wukv_full_tile_21.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_20 = {.shape = {1, 1, 1, 128} , .stride = out_part_18.stride, .addr = out_part_18.addr + ((0) * out_part_18.stride.c+(0) * out_part_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_18.stride.c+(0) * out_part_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_18.default_stride};
    __ppl_tensor_info out_cast_19 = {.shape = {1, 1, 1, 128} , .stride = out_cast_18.stride, .addr = out_cast_18.addr + ((0) * out_cast_18.stride.c+(0) * out_cast_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_18.stride.c+(0) * out_cast_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_18.default_stride};
    tpu_bdc_cast(out_cast_19.addr, out_part_20.addr, &out_cast_19.shape, (out_cast_19.default_stride ? NULL : &out_cast_19.stride), (out_part_20.default_stride ? NULL : &out_part_20.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_20 = {.shape = {1, 1, 1, 128} , .stride = out_cast_18.stride, .addr = out_cast_18.addr + ((0) * out_cast_18.stride.c+(0) * out_cast_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_18.stride.c+(0) * out_cast_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_18.default_stride};
    __ppl_tensor_info OUT_6 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(768) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(768) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_6.addr, out_cast_20.addr, &OUT_6.shape, (OUT_6.default_stride ? NULL : &OUT_6.stride), (out_cast_20.default_stride ? NULL : &out_cast_20.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_21 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_21.stride, 0, &q_nope_head_21.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_21 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_21.stride, 0, &q_wukv_acc_21.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_21 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_21.stride, 0, &q_wukv_head_21.shape, DT_BFP16);
    __ppl_tensor_info score_tile_14 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_14.stride, 0, &score_tile_14.shape, DT_FP32);
    __ppl_tensor_info prob_tile_14 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_14.stride, 0, &prob_tile_14.shape, DT_BFP16);
    __ppl_tensor_info scores_max_7 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_7.stride, 0, &scores_max_7.shape, DT_FP32);
    __ppl_tensor_info scores_sum_7 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_7.stride, 0, &scores_sum_7.shape, DT_FP32);
    __ppl_tensor_info work_score0_7 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_7.stride, 0, &work_score0_7.shape, DT_FP32);
    __ppl_tensor_info work_score1_7 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_7.stride, 0, &work_score1_7.shape, DT_FP32);
    __ppl_tensor_info coeff_7 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_7.stride, 0, &coeff_7.shape, DT_FP32);
    __ppl_tensor_info table_7 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_7.stride, 0, &table_7.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_14 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_14.stride, 0, &ctx_acc_14.shape, DT_FP32);
    __ppl_tensor_info ctx_head_21 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_21.stride, 0, &ctx_head_21.shape, DT_BFP16);
    __ppl_tensor_info out_part_21 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_21.stride, 0, &out_part_21.shape, DT_FP32);
    __ppl_tensor_info out_cast_21 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_21.stride, 0, &out_cast_21.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_15 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_12.stride, .addr = q_upper_h1_12.addr + ((0) * q_upper_h1_12.stride.c+(0) * q_upper_h1_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_12.stride.c+(0) * q_upper_h1_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_12.default_stride};
    __ppl_tensor_info q_nope_head_22 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_21.stride, .addr = q_nope_head_21.addr + ((0) * q_nope_head_21.stride.c+(0) * q_nope_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_21.stride.c+(0) * q_nope_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_21.default_stride};
    tpu_bdc_cpy(q_nope_head_22.addr, q_upper_h1_15.addr, &q_nope_head_22.shape, (q_nope_head_22.default_stride ? NULL : &q_nope_head_22.stride), (q_upper_h1_15.default_stride ? NULL : &q_upper_h1_15.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_21.addr, q_wukv_acc_21_scalar_f32, &q_wukv_acc_21.shape, (q_wukv_acc_21.default_stride ? NULL : &q_wukv_acc_21.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_23 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_21.stride, .addr = q_nope_head_21.addr + (((0) / 64) * q_nope_head_21.stride.c+(0) * q_nope_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_21.stride.c+(0) * q_nope_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_22 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_18.stride, .addr = wukv_full_tile_18.addr + (((256) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_22 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_21.stride, .addr = q_wukv_acc_21.addr + (((0) / 64) * q_wukv_acc_21.stride.c+(0) * q_wukv_acc_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_21.stride.c+(0) * q_wukv_acc_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_22.addr, q_nope_head_23.addr, wukv_full_tile_22.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_23 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_21.stride, .addr = q_wukv_acc_21.addr + ((0) * q_wukv_acc_21.stride.c+(0) * q_wukv_acc_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_21.stride.c+(0) * q_wukv_acc_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_21.default_stride};
    __ppl_tensor_info q_wukv_head_22 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_21.stride, .addr = q_wukv_head_21.addr + ((0) * q_wukv_head_21.stride.c+(0) * q_wukv_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_21.stride.c+(0) * q_wukv_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_21.default_stride};
    tpu_bdc_cast(q_wukv_head_22.addr, q_wukv_acc_23.addr, &q_wukv_head_22.shape, (q_wukv_head_22.default_stride ? NULL : &q_wukv_head_22.stride), (q_wukv_acc_23.default_stride ? NULL : &q_wukv_acc_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_23 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_21.stride, .addr = q_wukv_head_21.addr + (((0) / 64) * q_wukv_head_21.stride.c+(0) * q_wukv_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_21.stride.c+(0) * q_wukv_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_15 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_12.stride, .addr = kv_cache_full_12.addr + (((0) / 64) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_12.stride.c+(0) * kv_cache_full_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_11 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_9.stride, .addr = score_latent_h0_9.addr + (((0) / 64) * score_latent_h0_9.stride.c+(0) * score_latent_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_9.stride.c+(0) * score_latent_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_11.addr, q_wukv_head_23.addr, kv_cache_full_15.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_7 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_6.stride, .addr = q_rope_h1_6.addr + (((0) / 64) * q_rope_h1_6.stride.c+(0) * q_rope_h1_6.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_6.stride.c+(0) * q_rope_h1_6.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_15 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_12.stride, .addr = pe_score_tile_12.addr + (((0) / 64) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_12.stride.c+(0) * pe_score_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_11 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_9.stride, .addr = score_rope_h0_9.addr + (((0) / 64) * score_rope_h0_9.stride.c+(0) * score_rope_h0_9.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_9.stride.c+(0) * score_rope_h0_9.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_11.addr, q_rope_h1_7.addr, pe_score_tile_15.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_14.addr, score_latent_h0_9.addr, score_rope_h0_9.addr, &score_tile_14.shape, (score_tile_14.default_stride ? NULL : &score_tile_14.stride), (score_latent_h0_9.default_stride ? NULL : &score_latent_h0_9.stride), (score_rope_h0_9.default_stride ? NULL : &score_rope_h0_9.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_14.addr, score_tile_14.addr, mask_row_6.addr, &score_tile_14.shape, (score_tile_14.default_stride ? NULL : &score_tile_14.stride), (score_tile_14.default_stride ? NULL : &score_tile_14.stride), (mask_row_6.default_stride ? NULL : &mask_row_6.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_14.addr, score_tile_14.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_14.shape, (score_tile_14.default_stride ? NULL : &score_tile_14.stride), (score_tile_14.default_stride ? NULL : &score_tile_14.stride), DT_FP32);
    {
    scalar_t scores_max_7_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_7.addr, scores_max_7_scalar_f32, &scores_max_7.shape, (scores_max_7.default_stride ? NULL : &scores_max_7.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_7 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_7.stride, 0, &tmp_buffer_max_7.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_14.shape.n, score_tile_14.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_14.shape.n, score_tile_14.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_14.shape.n, score_tile_14.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_14.shape.n, score_tile_14.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_14.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, score_tile_14.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_14.shape.w) {
          dim4 fill_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, align_w - score_tile_14.shape.w};
          int elem_size = 4;
          int offset = score_tile_14.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_14.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_14.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_7_stride;
    tpu_aligned_stride(&scores_max_7_stride, 0, &scores_max_7.shape, DT_FP32);
    scores_max_7_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_14.addr, score_tile_14.addr, scores_max_7.addr, &score_tile_14.shape, (score_tile_14.default_stride ? NULL : &score_tile_14.stride), (score_tile_14.default_stride ? NULL : &score_tile_14.stride), &scores_max_7_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_7.addr);
    tpu_bdc_load_fp32_exp_table(table_7.addr);
    tpu_bdc_fp32_exp(score_tile_14.addr, score_tile_14.addr, work_score0_7.addr, work_score1_7.addr, coeff_7.addr, table_7.addr, &score_tile_14.shape);
    __ppl_tensor_info tmp_buffer_sum_7 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_7.stride, 0, &tmp_buffer_sum_7.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_14.shape.n, score_tile_14.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_14.shape.n, score_tile_14.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_14.shape.n, score_tile_14.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_14.shape.n, score_tile_14.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_14.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, score_tile_14.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_14.shape.w) {
          dim4 fill_shape = {score_tile_14.shape.n, score_tile_14.shape.c, 1, align_w - score_tile_14.shape.w};
          int elem_size = 4;
          int offset = score_tile_14.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_14.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_14.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_7.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_15 = {.shape = {1, 1, 1, 192} , .stride = score_tile_14.stride, .addr = score_tile_14.addr + ((0) * score_tile_14.stride.c+(0) * score_tile_14.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_14.stride.c+(0) * score_tile_14.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_14.default_stride};
    __ppl_tensor_info prob_tile_15 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_14.stride, .addr = prob_tile_14.addr + ((0) * prob_tile_14.stride.c+(0) * prob_tile_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_14.stride.c+(0) * prob_tile_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_14.default_stride};
    tpu_bdc_cast(prob_tile_15.addr, score_tile_15.addr, &prob_tile_15.shape, (prob_tile_15.default_stride ? NULL : &prob_tile_15.stride), (score_tile_15.default_stride ? NULL : &score_tile_15.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_14_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_14.addr, ctx_acc_14_scalar_f32, &ctx_acc_14.shape, (ctx_acc_14.default_stride ? NULL : &ctx_acc_14.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_14.addr, prob_tile_14.addr, kv_cache_full_12.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_7_stride;
    tpu_aligned_stride(&scores_sum_7_stride, 0, &scores_sum_7.shape, DT_FP32);
    scores_sum_7_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_14.addr, ctx_acc_14.addr, scores_sum_7.addr, &ctx_acc_14.shape, (ctx_acc_14.default_stride ? NULL : &ctx_acc_14.stride), (ctx_acc_14.default_stride ? NULL : &ctx_acc_14.stride), &scores_sum_7_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_15 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_14.stride, .addr = ctx_acc_14.addr + ((0) * ctx_acc_14.stride.c+(0) * ctx_acc_14.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_14.stride.c+(0) * ctx_acc_14.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_14.default_stride};
    __ppl_tensor_info ctx_head_22 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_21.stride, .addr = ctx_head_21.addr + ((0) * ctx_head_21.stride.c+(0) * ctx_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_21.stride.c+(0) * ctx_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_21.default_stride};
    tpu_bdc_cast(ctx_head_22.addr, ctx_acc_15.addr, &ctx_head_22.shape, (ctx_head_22.default_stride ? NULL : &ctx_head_22.stride), (ctx_acc_15.default_stride ? NULL : &ctx_acc_15.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_21.addr, out_part_21_scalar_f32, &out_part_21.shape, (out_part_21.default_stride ? NULL : &out_part_21.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_23 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_21.stride, .addr = ctx_head_21.addr + (((0) / 64) * ctx_head_21.stride.c+(0) * ctx_head_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_21.stride.c+(0) * ctx_head_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_23 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_18.stride, .addr = wukv_full_tile_18.addr + (((384) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_18.stride.c+(0) * wukv_full_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_22 = {.shape = {1, 1, 1, 128}, .stride = out_part_21.stride, .addr = out_part_21.addr + (((0) / 64) * out_part_21.stride.c+(0) * out_part_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_21.stride.c+(0) * out_part_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_22.addr, ctx_head_23.addr, wukv_full_tile_23.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_23 = {.shape = {1, 1, 1, 128} , .stride = out_part_21.stride, .addr = out_part_21.addr + ((0) * out_part_21.stride.c+(0) * out_part_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_21.stride.c+(0) * out_part_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_21.default_stride};
    __ppl_tensor_info out_cast_22 = {.shape = {1, 1, 1, 128} , .stride = out_cast_21.stride, .addr = out_cast_21.addr + ((0) * out_cast_21.stride.c+(0) * out_cast_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_21.stride.c+(0) * out_cast_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_21.default_stride};
    tpu_bdc_cast(out_cast_22.addr, out_part_23.addr, &out_cast_22.shape, (out_cast_22.default_stride ? NULL : &out_cast_22.stride), (out_part_23.default_stride ? NULL : &out_part_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_23 = {.shape = {1, 1, 1, 128} , .stride = out_cast_21.stride, .addr = out_cast_21.addr + ((0) * out_cast_21.stride.c+(0) * out_cast_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_21.stride.c+(0) * out_cast_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_21.default_stride};
    __ppl_tensor_info OUT_7 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(896) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(896) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_7.addr, out_cast_23.addr, &OUT_7.shape, (OUT_7.default_stride ? NULL : &OUT_7.stride), (out_cast_23.default_stride ? NULL : &out_cast_23.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_12.stride, 0, &q_upper_acc_h0_12.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_16 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_16.stride, 0, &q_upper_h0_16.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_12.stride, 0, &q_upper_acc_h1_12.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_16 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_16.stride, 0, &q_upper_h1_16.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_8.stride, 0, &q_rope_src_h0_8.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_8.stride, 0, &q_rope_h0_8.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_4.stride, 0, &q_rope_tmp0_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_4.stride, 0, &q_rope_tmp1_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_4.stride, 0, &q_rope_tmp2_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_4.stride, 0, &q_rope_tmp3_h0_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_8.stride, 0, &q_rope_src_h1_8.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_8.stride, 0, &q_rope_h1_8.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_4.stride, 0, &q_rope_tmp0_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_4.stride, 0, &q_rope_tmp1_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_4.stride, 0, &q_rope_tmp2_h1_4.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_4.stride, 0, &q_rope_tmp3_h1_4.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_8.stride, 0, &cos_shared_8.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_8.stride, 0, &sin_shared_8.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_8.stride, 0, &pe_cur_8.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_4.stride, 0, &pe_cur_cos_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_4.stride, 0, &pe_cur_sin_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_4.stride, 0, &pe_cur_neg_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_4 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_4.stride, 0, &pe_cur_neg_sin_4.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_8 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_8.stride, 0, &pe_cur_rope_8.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_24 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_24.stride, 0, &wukv_full_tile_24.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_16 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_16.stride, 0, &kv_cache_full_16.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_16 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_16.stride, 0, &pe_score_tile_16.shape, DT_BFP16);
  __ppl_tensor_info mask_row_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_8.stride, 0, &mask_row_8.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_12.stride, 0, &score_latent_h0_12.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_12.stride, 0, &score_rope_h0_12.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_4.stride, 0, &score_latent_h1_4.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_4 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_4.stride, 0, &score_rope_h1_4.shape, DT_FP32);
  if (tpu_workitem_index() == 4) {
    __ppl_tensor_info COS_4 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_9 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_8.stride, .addr = cos_shared_8.addr + ((0) * cos_shared_8.stride.c+(0) * cos_shared_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_8.stride.c+(0) * cos_shared_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_8.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_9.addr, COS_4.addr, &cos_shared_9.shape, (cos_shared_9.default_stride ? NULL : &cos_shared_9.stride), (COS_4.default_stride ? NULL : &COS_4.stride), DT_BFP16);
    __ppl_tensor_info SIN_4 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_9 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_8.stride, .addr = sin_shared_8.addr + ((0) * sin_shared_8.stride.c+(0) * sin_shared_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_8.stride.c+(0) * sin_shared_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_8.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_9.addr, SIN_4.addr, &sin_shared_9.shape, (sin_shared_9.default_stride ? NULL : &sin_shared_9.stride), (SIN_4.default_stride ? NULL : &SIN_4.stride), DT_BFP16);
    for (int d_8 = 0; d_8 < 4; ++d_8) {
      __ppl_tensor_info kv_cur_tile_12 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_12.stride, 0, &kv_cur_tile_12.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_4 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_8 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_8 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_13 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_12.stride, .addr = kv_cur_tile_12.addr + ((0) * kv_cur_tile_12.stride.c+(0) * kv_cur_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_12.stride.c+(0) * kv_cur_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_12.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_13.addr, KV_CUR_4.addr, &kv_cur_tile_13.shape, (kv_cur_tile_13.default_stride ? NULL : &kv_cur_tile_13.stride), (KV_CUR_4.default_stride ? NULL : &KV_CUR_4.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_14 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_12.stride, .addr = kv_cur_tile_12.addr + ((0) * kv_cur_tile_12.stride.c+(0) * kv_cur_tile_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_12.stride.c+(0) * kv_cur_tile_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_12.default_stride};
      __ppl_tensor_info KVcache_OUT_8 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_8 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_8 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_8.addr, kv_cur_tile_14.addr, &KVcache_OUT_8.shape, (KVcache_OUT_8.default_stride ? NULL : &KVcache_OUT_8.stride), (kv_cur_tile_14.default_stride ? NULL : &kv_cur_tile_14.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_4 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_9 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_8.stride, .addr = pe_cur_8.addr + ((0) * pe_cur_8.stride.c+(0) * pe_cur_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_8.stride.c+(0) * pe_cur_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_8.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_9.addr, PE_CUR_4.addr, &pe_cur_9.shape, (pe_cur_9.default_stride ? NULL : &pe_cur_9.stride), (PE_CUR_4.default_stride ? NULL : &PE_CUR_4.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_4.addr, pe_cur_8.addr, cos_shared_8.addr, &pe_cur_cos_4.shape, (pe_cur_cos_4.default_stride ? NULL : &pe_cur_cos_4.stride), (pe_cur_8.default_stride ? NULL : &pe_cur_8.stride), (cos_shared_8.default_stride ? NULL : &cos_shared_8.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_4.addr, pe_cur_8.addr, sin_shared_8.addr, &pe_cur_sin_4.shape, (pe_cur_sin_4.default_stride ? NULL : &pe_cur_sin_4.stride), (pe_cur_8.default_stride ? NULL : &pe_cur_8.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_4_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_4_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_4_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_4.addr, pe_cur_8.addr, pe_cur_neg_4_scalar_DT_BFP16, &pe_cur_neg_4.shape, (pe_cur_neg_4.default_stride ? NULL : &pe_cur_neg_4.stride), (pe_cur_8.default_stride ? NULL : &pe_cur_8.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_4.addr, pe_cur_neg_4.addr, sin_shared_8.addr, &pe_cur_neg_sin_4.shape, (pe_cur_neg_sin_4.default_stride ? NULL : &pe_cur_neg_sin_4.stride), (pe_cur_neg_4.default_stride ? NULL : &pe_cur_neg_4.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_8.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_8.shape.n, .c = pe_cur_rope_8.shape.c, .h = pe_cur_rope_8.shape.h, .w = pe_cur_rope_8.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_8.addr, pe_cur_cos_4.addr, pe_cur_neg_sin_4.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_8.addr + 2, pe_cur_cos_4.addr + 2, pe_cur_sin_4.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_9 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_8.stride, .addr = pe_cur_rope_8.addr + ((0) * pe_cur_rope_8.stride.c+(0) * pe_cur_rope_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_8.stride.c+(0) * pe_cur_rope_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_8.default_stride};
    __ppl_tensor_info PEcache_OUT_8 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_8.addr, pe_cur_rope_9.addr, &PEcache_OUT_8.shape, (PEcache_OUT_8.default_stride ? NULL : &PEcache_OUT_8.stride), (pe_cur_rope_9.default_stride ? NULL : &pe_cur_rope_9.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_12.addr, q_upper_acc_h0_12_scalar_f32, &q_upper_acc_h0_12.shape, (q_upper_acc_h0_12.default_stride ? NULL : &q_upper_acc_h0_12.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_12.addr, q_upper_acc_h1_12_scalar_f32, &q_upper_acc_h1_12.shape, (q_upper_acc_h1_12.default_stride ? NULL : &q_upper_acc_h1_12.stride), DT_FP32);
    }
    for (int k_4 = 0; k_4 < 12; ++k_4) {
      __ppl_tensor_info q_tile_16 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_16.stride, 0, &q_tile_16.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_12 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_12.stride, 0, &wuq_fp8_12.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_32 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_32.stride, 0, &wuq_scale_stage_32.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_16 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_16.stride, 0, &wuq_tile_16.shape, DT_BFP16);
      __ppl_tensor_info Q_4 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_4 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_4 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_17 = {.shape = {1, 1, 1, 128} , .stride = q_tile_16.stride, .addr = q_tile_16.addr + ((0) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_16.default_stride};
      tpu_gdma_cpy_S2L(q_tile_17.addr, Q_4.addr, &q_tile_17.shape, (q_tile_17.default_stride ? NULL : &q_tile_17.stride), (Q_4.default_stride ? NULL : &Q_4.stride), DT_BFP16);
      __ppl_tensor_info WUQ_4 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((1536) * 1536+((k_4 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((1536) * 1536+((k_4 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_13 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_12.stride, .addr = wuq_fp8_12.addr + ((0) * wuq_fp8_12.stride.c+(0) * wuq_fp8_12.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_12.stride.c+(0) * wuq_fp8_12.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_12.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_13.addr, WUQ_4.addr, &wuq_fp8_13.shape, (wuq_fp8_13.default_stride ? NULL : &wuq_fp8_13.stride), (WUQ_4.default_stride ? NULL : &WUQ_4.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_72 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_72.stride, 0, &wuq_scale_scalar_72.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_24 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((12) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((12) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_73 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_72.stride, .addr = wuq_scale_scalar_72.addr + ((0) * wuq_scale_scalar_72.stride.c+(0) * wuq_scale_scalar_72.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_72.stride.c+(0) * wuq_scale_scalar_72.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_72.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_73.addr, WUQ_SCALE_24.addr, &wuq_scale_scalar_73.shape, (wuq_scale_scalar_73.default_stride ? NULL : &wuq_scale_scalar_73.stride), (WUQ_SCALE_24.default_stride ? NULL : &WUQ_SCALE_24.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_74 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_72.stride, .addr = wuq_scale_scalar_72.addr + (((0) / 64) * wuq_scale_scalar_72.stride.c+(0) * wuq_scale_scalar_72.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_72.stride.c+(0) * wuq_scale_scalar_72.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_33 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((0) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_33.addr, wuq_scale_scalar_74.addr, &wuq_scale_stage_33.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_75 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_75.stride, 0, &wuq_scale_scalar_75.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_25 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((12) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((12) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_76 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_75.stride, .addr = wuq_scale_scalar_75.addr + ((0) * wuq_scale_scalar_75.stride.c+(0) * wuq_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_75.stride.c+(0) * wuq_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_75.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_76.addr, WUQ_SCALE_25.addr, &wuq_scale_scalar_76.shape, (wuq_scale_scalar_76.default_stride ? NULL : &wuq_scale_scalar_76.stride), (WUQ_SCALE_25.default_stride ? NULL : &WUQ_SCALE_25.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_77 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_75.stride, .addr = wuq_scale_scalar_75.addr + (((0) / 64) * wuq_scale_scalar_75.stride.c+(0) * wuq_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_75.stride.c+(0) * wuq_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_34 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((64) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_34.addr, wuq_scale_scalar_77.addr, &wuq_scale_stage_34.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_78 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_78.stride, 0, &wuq_scale_scalar_78.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_26 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((13) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((13) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_79 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_78.stride, .addr = wuq_scale_scalar_78.addr + ((0) * wuq_scale_scalar_78.stride.c+(0) * wuq_scale_scalar_78.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_78.stride.c+(0) * wuq_scale_scalar_78.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_78.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_79.addr, WUQ_SCALE_26.addr, &wuq_scale_scalar_79.shape, (wuq_scale_scalar_79.default_stride ? NULL : &wuq_scale_scalar_79.stride), (WUQ_SCALE_26.default_stride ? NULL : &WUQ_SCALE_26.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_80 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_78.stride, .addr = wuq_scale_scalar_78.addr + (((0) / 64) * wuq_scale_scalar_78.stride.c+(0) * wuq_scale_scalar_78.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_78.stride.c+(0) * wuq_scale_scalar_78.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_35 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((128) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_35.addr, wuq_scale_scalar_80.addr, &wuq_scale_stage_35.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_81 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_81.stride, 0, &wuq_scale_scalar_81.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_27 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((13) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((13) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_82 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_81.stride, .addr = wuq_scale_scalar_81.addr + ((0) * wuq_scale_scalar_81.stride.c+(0) * wuq_scale_scalar_81.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_81.stride.c+(0) * wuq_scale_scalar_81.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_81.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_82.addr, WUQ_SCALE_27.addr, &wuq_scale_scalar_82.shape, (wuq_scale_scalar_82.default_stride ? NULL : &wuq_scale_scalar_82.stride), (WUQ_SCALE_27.default_stride ? NULL : &WUQ_SCALE_27.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_83 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_81.stride, .addr = wuq_scale_scalar_81.addr + (((0) / 64) * wuq_scale_scalar_81.stride.c+(0) * wuq_scale_scalar_81.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_81.stride.c+(0) * wuq_scale_scalar_81.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_36 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((192) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_36.addr, wuq_scale_scalar_83.addr, &wuq_scale_stage_36.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_84 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_84.stride, 0, &wuq_scale_scalar_84.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_28 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((14) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((14) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_85 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_84.stride, .addr = wuq_scale_scalar_84.addr + ((0) * wuq_scale_scalar_84.stride.c+(0) * wuq_scale_scalar_84.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_84.stride.c+(0) * wuq_scale_scalar_84.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_84.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_85.addr, WUQ_SCALE_28.addr, &wuq_scale_scalar_85.shape, (wuq_scale_scalar_85.default_stride ? NULL : &wuq_scale_scalar_85.stride), (WUQ_SCALE_28.default_stride ? NULL : &WUQ_SCALE_28.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_86 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_84.stride, .addr = wuq_scale_scalar_84.addr + (((0) / 64) * wuq_scale_scalar_84.stride.c+(0) * wuq_scale_scalar_84.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_84.stride.c+(0) * wuq_scale_scalar_84.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_37 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((256) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_37.addr, wuq_scale_scalar_86.addr, &wuq_scale_stage_37.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_87 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_87.stride, 0, &wuq_scale_scalar_87.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_29 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((14) * 12+(k_4) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((14) * 12+(k_4) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_88 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_87.stride, .addr = wuq_scale_scalar_87.addr + ((0) * wuq_scale_scalar_87.stride.c+(0) * wuq_scale_scalar_87.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_87.stride.c+(0) * wuq_scale_scalar_87.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_87.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_88.addr, WUQ_SCALE_29.addr, &wuq_scale_scalar_88.shape, (wuq_scale_scalar_88.default_stride ? NULL : &wuq_scale_scalar_88.stride), (WUQ_SCALE_29.default_stride ? NULL : &WUQ_SCALE_29.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_89 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_87.stride, .addr = wuq_scale_scalar_87.addr + (((0) / 64) * wuq_scale_scalar_87.stride.c+(0) * wuq_scale_scalar_87.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_87.stride.c+(0) * wuq_scale_scalar_87.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_38 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((320) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_38.addr, wuq_scale_scalar_89.addr, &wuq_scale_stage_38.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_14 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_12.stride, .addr = wuq_fp8_12.addr + (((0) / 64) * wuq_fp8_12.stride.c+(0) * wuq_fp8_12.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_12.stride.c+(0) * wuq_fp8_12.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_39 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_32.stride, .addr = wuq_scale_stage_32.addr + (((0) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_32.stride.c+(0) * wuq_scale_stage_32.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_17 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_16.stride, .addr = wuq_tile_16.addr + (((0) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_17.addr, wuq_fp8_14.addr, &wuq_tile_17.shape, (wuq_tile_17.default_stride ? NULL : &wuq_tile_17.stride), (wuq_fp8_14.default_stride ? NULL : &wuq_fp8_14.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_39_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_39_block_stride, 0, &wuq_scale_stage_39.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_39_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_17.addr, wuq_tile_17.addr, wuq_scale_stage_39.addr, &wuq_tile_17.shape, (wuq_tile_17.default_stride ? NULL : &wuq_tile_17.stride), (wuq_tile_17.default_stride ? NULL : &wuq_tile_17.stride), &wuq_scale_stage_39_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_18 = {.shape = {1, 1, 1, 128}, .stride = q_tile_16.stride, .addr = q_tile_16.addr + (((0) / 64) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_18 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_16.stride, .addr = wuq_tile_16.addr + (((0) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_13 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_12.stride, .addr = q_upper_acc_h0_12.addr + (((0) / 64) * q_upper_acc_h0_12.stride.c+(0) * q_upper_acc_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_12.stride.c+(0) * q_upper_acc_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_13.addr, q_tile_18.addr, wuq_tile_18.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_19 = {.shape = {1, 1, 1, 128}, .stride = q_tile_16.stride, .addr = q_tile_16.addr + (((0) / 64) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_16.stride.c+(0) * q_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_19 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_16.stride, .addr = wuq_tile_16.addr + (((192) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_16.stride.c+(0) * wuq_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_13 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_12.stride, .addr = q_upper_acc_h1_12.addr + (((0) / 64) * q_upper_acc_h1_12.stride.c+(0) * q_upper_acc_h1_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_12.stride.c+(0) * q_upper_acc_h1_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_13.addr, q_tile_19.addr, wuq_tile_19.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_14 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_12.stride, .addr = q_upper_acc_h0_12.addr + ((0) * q_upper_acc_h0_12.stride.c+(0) * q_upper_acc_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_12.stride.c+(0) * q_upper_acc_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_12.default_stride};
    __ppl_tensor_info q_upper_h0_17 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_16.stride, .addr = q_upper_h0_16.addr + ((0) * q_upper_h0_16.stride.c+(0) * q_upper_h0_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_16.stride.c+(0) * q_upper_h0_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_16.default_stride};
    tpu_bdc_cast(q_upper_h0_17.addr, q_upper_acc_h0_14.addr, &q_upper_h0_17.shape, (q_upper_h0_17.default_stride ? NULL : &q_upper_h0_17.stride), (q_upper_acc_h0_14.default_stride ? NULL : &q_upper_acc_h0_14.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_14 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_12.stride, .addr = q_upper_acc_h1_12.addr + ((0) * q_upper_acc_h1_12.stride.c+(0) * q_upper_acc_h1_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_12.stride.c+(0) * q_upper_acc_h1_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_12.default_stride};
    __ppl_tensor_info q_upper_h1_17 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_16.stride, .addr = q_upper_h1_16.addr + ((0) * q_upper_h1_16.stride.c+(0) * q_upper_h1_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_16.stride.c+(0) * q_upper_h1_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_16.default_stride};
    tpu_bdc_cast(q_upper_h1_17.addr, q_upper_acc_h1_14.addr, &q_upper_h1_17.shape, (q_upper_h1_17.default_stride ? NULL : &q_upper_h1_17.stride), (q_upper_acc_h1_14.default_stride ? NULL : &q_upper_acc_h1_14.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_9 = 0; d_9 < 4; ++d_9) {
      __ppl_tensor_info wukv_fp8_12 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_12.stride, 0, &wukv_fp8_12.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_24 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_24.stride, 0, &wukv_scale_stage_24.shape, DT_BFP16);
      __ppl_tensor_info WUKV_4 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((2048) * 512+((d_9 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((2048) * 512+((d_9 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_13 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_12.stride, .addr = wukv_fp8_12.addr + ((0) * wukv_fp8_12.stride.c+(0) * wukv_fp8_12.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_12.stride.c+(0) * wukv_fp8_12.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_12.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_13.addr, WUKV_4.addr, &wukv_fp8_13.shape, (wukv_fp8_13.default_stride ? NULL : &wukv_fp8_13.stride), (WUKV_4.default_stride ? NULL : &WUKV_4.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_40 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_40.stride, 0, &wukv_scale_scalar_40.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_16 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((16) * 4+(d_9) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((16) * 4+(d_9) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_41 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_40.stride, .addr = wukv_scale_scalar_40.addr + ((0) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_40.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_41.addr, WUKV_SCALE_16.addr, &wukv_scale_scalar_41.shape, (wukv_scale_scalar_41.default_stride ? NULL : &wukv_scale_scalar_41.stride), (WUKV_SCALE_16.default_stride ? NULL : &WUKV_SCALE_16.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_42 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_40.stride, .addr = wukv_scale_scalar_40.addr + (((0) / 64) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_25 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_24.stride, .addr = wukv_scale_stage_24.addr + (((0) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_25_chunk = wukv_scale_stage_25;
      wukv_scale_stage_25_chunk.shape.c = 64;
      wukv_scale_stage_25_chunk.addr = wukv_scale_stage_25.addr + ((0) * wukv_scale_stage_25.stride.c) * 2;
      wukv_scale_stage_25_chunk.offset = wukv_scale_stage_25.offset + ((0) * wukv_scale_stage_25.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_25_chunk.addr, wukv_scale_scalar_42.addr, &wukv_scale_stage_25_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_25_chunk_1 = wukv_scale_stage_25;
      wukv_scale_stage_25_chunk_1.shape.c = 64;
      wukv_scale_stage_25_chunk_1.addr = wukv_scale_stage_25.addr + ((1) * wukv_scale_stage_25.stride.c) * 2;
      wukv_scale_stage_25_chunk_1.offset = wukv_scale_stage_25.offset + ((1) * wukv_scale_stage_25.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_25_chunk_1.addr, wukv_scale_scalar_42.addr, &wukv_scale_stage_25_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_17 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((17) * 4+(d_9) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((17) * 4+(d_9) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_43 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_40.stride, .addr = wukv_scale_scalar_40.addr + ((0) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_40.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_43.addr, WUKV_SCALE_17.addr, &wukv_scale_scalar_43.shape, (wukv_scale_scalar_43.default_stride ? NULL : &wukv_scale_scalar_43.stride), (WUKV_SCALE_17.default_stride ? NULL : &WUKV_SCALE_17.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_44 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_40.stride, .addr = wukv_scale_scalar_40.addr + (((0) / 64) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_40.stride.c+(0) * wukv_scale_scalar_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_26 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_24.stride, .addr = wukv_scale_stage_24.addr + (((128) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_26_chunk = wukv_scale_stage_26;
      wukv_scale_stage_26_chunk.shape.c = 64;
      wukv_scale_stage_26_chunk.addr = wukv_scale_stage_26.addr + ((0) * wukv_scale_stage_26.stride.c) * 2;
      wukv_scale_stage_26_chunk.offset = wukv_scale_stage_26.offset + ((0) * wukv_scale_stage_26.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_26_chunk.addr, wukv_scale_scalar_44.addr, &wukv_scale_stage_26_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_26_chunk_1 = wukv_scale_stage_26;
      wukv_scale_stage_26_chunk_1.shape.c = 64;
      wukv_scale_stage_26_chunk_1.addr = wukv_scale_stage_26.addr + ((1) * wukv_scale_stage_26.stride.c) * 2;
      wukv_scale_stage_26_chunk_1.offset = wukv_scale_stage_26.offset + ((1) * wukv_scale_stage_26.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_26_chunk_1.addr, wukv_scale_scalar_44.addr, &wukv_scale_stage_26_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_45 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_45.stride, 0, &wukv_scale_scalar_45.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_18 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((18) * 4+(d_9) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((18) * 4+(d_9) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_46 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_45.stride, .addr = wukv_scale_scalar_45.addr + ((0) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_45.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_46.addr, WUKV_SCALE_18.addr, &wukv_scale_scalar_46.shape, (wukv_scale_scalar_46.default_stride ? NULL : &wukv_scale_scalar_46.stride), (WUKV_SCALE_18.default_stride ? NULL : &WUKV_SCALE_18.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_47 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_45.stride, .addr = wukv_scale_scalar_45.addr + (((0) / 64) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_27 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_24.stride, .addr = wukv_scale_stage_24.addr + (((256) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_27_chunk = wukv_scale_stage_27;
      wukv_scale_stage_27_chunk.shape.c = 64;
      wukv_scale_stage_27_chunk.addr = wukv_scale_stage_27.addr + ((0) * wukv_scale_stage_27.stride.c) * 2;
      wukv_scale_stage_27_chunk.offset = wukv_scale_stage_27.offset + ((0) * wukv_scale_stage_27.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_27_chunk.addr, wukv_scale_scalar_47.addr, &wukv_scale_stage_27_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_27_chunk_1 = wukv_scale_stage_27;
      wukv_scale_stage_27_chunk_1.shape.c = 64;
      wukv_scale_stage_27_chunk_1.addr = wukv_scale_stage_27.addr + ((1) * wukv_scale_stage_27.stride.c) * 2;
      wukv_scale_stage_27_chunk_1.offset = wukv_scale_stage_27.offset + ((1) * wukv_scale_stage_27.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_27_chunk_1.addr, wukv_scale_scalar_47.addr, &wukv_scale_stage_27_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_19 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((19) * 4+(d_9) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((19) * 4+(d_9) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_48 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_45.stride, .addr = wukv_scale_scalar_45.addr + ((0) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_45.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_48.addr, WUKV_SCALE_19.addr, &wukv_scale_scalar_48.shape, (wukv_scale_scalar_48.default_stride ? NULL : &wukv_scale_scalar_48.stride), (WUKV_SCALE_19.default_stride ? NULL : &WUKV_SCALE_19.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_49 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_45.stride, .addr = wukv_scale_scalar_45.addr + (((0) / 64) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_45.stride.c+(0) * wukv_scale_scalar_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_28 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_24.stride, .addr = wukv_scale_stage_24.addr + (((384) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_28_chunk = wukv_scale_stage_28;
      wukv_scale_stage_28_chunk.shape.c = 64;
      wukv_scale_stage_28_chunk.addr = wukv_scale_stage_28.addr + ((0) * wukv_scale_stage_28.stride.c) * 2;
      wukv_scale_stage_28_chunk.offset = wukv_scale_stage_28.offset + ((0) * wukv_scale_stage_28.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_28_chunk.addr, wukv_scale_scalar_49.addr, &wukv_scale_stage_28_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_28_chunk_1 = wukv_scale_stage_28;
      wukv_scale_stage_28_chunk_1.shape.c = 64;
      wukv_scale_stage_28_chunk_1.addr = wukv_scale_stage_28.addr + ((1) * wukv_scale_stage_28.stride.c) * 2;
      wukv_scale_stage_28_chunk_1.offset = wukv_scale_stage_28.offset + ((1) * wukv_scale_stage_28.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_28_chunk_1.addr, wukv_scale_scalar_49.addr, &wukv_scale_stage_28_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_14 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_12.stride, .addr = wukv_fp8_12.addr + (((0) / 64) * wukv_fp8_12.stride.c+(0) * wukv_fp8_12.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_12.stride.c+(0) * wukv_fp8_12.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_29 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_24.stride, .addr = wukv_scale_stage_24.addr + (((0) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_24.stride.c+(0) * wukv_scale_stage_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_25 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_24.stride, .addr = wukv_full_tile_24.addr + (((0) / 64) * wukv_full_tile_24.stride.c+((d_9 * 128)) * wukv_full_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_24.stride.c+((d_9 * 128)) * wukv_full_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_25.addr, wukv_fp8_14.addr, &wukv_full_tile_25.shape, (wukv_full_tile_25.default_stride ? NULL : &wukv_full_tile_25.stride), (wukv_fp8_14.default_stride ? NULL : &wukv_fp8_14.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_29_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_29_block_stride, 0, &wukv_scale_stage_29.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_29_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_25.addr, wukv_full_tile_25.addr, wukv_scale_stage_29.addr, &wukv_full_tile_25.shape, (wukv_full_tile_25.default_stride ? NULL : &wukv_full_tile_25.stride), (wukv_full_tile_25.default_stride ? NULL : &wukv_full_tile_25.stride), &wukv_scale_stage_29_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_9 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_17 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_16.stride, .addr = kv_cache_full_16.addr + ((0) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_16.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_17.addr, KVcache_OUT_9.addr, &kv_cache_full_17.shape, (kv_cache_full_17.default_stride ? NULL : &kv_cache_full_17.stride), (KVcache_OUT_9.default_stride ? NULL : &KVcache_OUT_9.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_9 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_17 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_16.stride, .addr = pe_score_tile_16.addr + ((0) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_16.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_17.addr, PEcache_OUT_9.addr, &pe_score_tile_17.shape, (pe_score_tile_17.default_stride ? NULL : &pe_score_tile_17.stride), (PEcache_OUT_9.default_stride ? NULL : &PEcache_OUT_9.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_4 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_9 = {.shape = {1, 1, 1, 192} , .stride = mask_row_8.stride, .addr = mask_row_8.addr + ((0) * mask_row_8.stride.c+(0) * mask_row_8.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_8.stride.c+(0) * mask_row_8.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_8.default_stride};
    tpu_gdma_cpy_S2L(mask_row_9.addr, DECODE_MASK_4.addr, &mask_row_9.shape, (mask_row_9.default_stride ? NULL : &mask_row_9.stride), (DECODE_MASK_4.default_stride ? NULL : &DECODE_MASK_4.stride), DT_FP32);
    {
    scalar_t score_latent_h0_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_12.addr, score_latent_h0_12_scalar_f32, &score_latent_h0_12.shape, (score_latent_h0_12.default_stride ? NULL : &score_latent_h0_12.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_12_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_12.addr, score_rope_h0_12_scalar_f32, &score_rope_h0_12.shape, (score_rope_h0_12.default_stride ? NULL : &score_rope_h0_12.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_4_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_4.addr, score_latent_h1_4_scalar_f32, &score_latent_h1_4.shape, (score_latent_h1_4.default_stride ? NULL : &score_latent_h1_4.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_4_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_4.addr, score_rope_h1_4_scalar_f32, &score_rope_h1_4.shape, (score_rope_h1_4.default_stride ? NULL : &score_rope_h1_4.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_18 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_16.stride, .addr = q_upper_h0_16.addr + ((0) * q_upper_h0_16.stride.c+(128) * q_upper_h0_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_16.stride.c+(128) * q_upper_h0_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_16.default_stride};
    __ppl_tensor_info q_rope_src_h0_9 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_8.stride, .addr = q_rope_src_h0_8.addr + ((0) * q_rope_src_h0_8.stride.c+(0) * q_rope_src_h0_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_8.stride.c+(0) * q_rope_src_h0_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_8.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_9.addr, q_upper_h0_18.addr, &q_rope_src_h0_9.shape, (q_rope_src_h0_9.default_stride ? NULL : &q_rope_src_h0_9.stride), (q_upper_h0_18.default_stride ? NULL : &q_upper_h0_18.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_4.addr, q_rope_src_h0_8.addr, cos_shared_8.addr, &q_rope_tmp0_h0_4.shape, (q_rope_tmp0_h0_4.default_stride ? NULL : &q_rope_tmp0_h0_4.stride), (q_rope_src_h0_8.default_stride ? NULL : &q_rope_src_h0_8.stride), (cos_shared_8.default_stride ? NULL : &cos_shared_8.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_4.addr, q_rope_src_h0_8.addr, sin_shared_8.addr, &q_rope_tmp1_h0_4.shape, (q_rope_tmp1_h0_4.default_stride ? NULL : &q_rope_tmp1_h0_4.stride), (q_rope_src_h0_8.default_stride ? NULL : &q_rope_src_h0_8.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_4_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_4_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_4_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_4.addr, q_rope_src_h0_8.addr, q_rope_tmp2_h0_4_scalar_DT_BFP16, &q_rope_tmp2_h0_4.shape, (q_rope_tmp2_h0_4.default_stride ? NULL : &q_rope_tmp2_h0_4.stride), (q_rope_src_h0_8.default_stride ? NULL : &q_rope_src_h0_8.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_4.addr, q_rope_tmp2_h0_4.addr, sin_shared_8.addr, &q_rope_tmp3_h0_4.shape, (q_rope_tmp3_h0_4.default_stride ? NULL : &q_rope_tmp3_h0_4.stride), (q_rope_tmp2_h0_4.default_stride ? NULL : &q_rope_tmp2_h0_4.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_8.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_8.shape.n, .c = q_rope_h0_8.shape.c, .h = q_rope_h0_8.shape.h, .w = q_rope_h0_8.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_8.addr, q_rope_tmp0_h0_4.addr, q_rope_tmp3_h0_4.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_8.addr + 2, q_rope_tmp0_h0_4.addr + 2, q_rope_tmp1_h0_4.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_18 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_16.stride, .addr = q_upper_h1_16.addr + ((0) * q_upper_h1_16.stride.c+(128) * q_upper_h1_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_16.stride.c+(128) * q_upper_h1_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_16.default_stride};
    __ppl_tensor_info q_rope_src_h1_9 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_8.stride, .addr = q_rope_src_h1_8.addr + ((0) * q_rope_src_h1_8.stride.c+(0) * q_rope_src_h1_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_8.stride.c+(0) * q_rope_src_h1_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_8.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_9.addr, q_upper_h1_18.addr, &q_rope_src_h1_9.shape, (q_rope_src_h1_9.default_stride ? NULL : &q_rope_src_h1_9.stride), (q_upper_h1_18.default_stride ? NULL : &q_upper_h1_18.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_4.addr, q_rope_src_h1_8.addr, cos_shared_8.addr, &q_rope_tmp0_h1_4.shape, (q_rope_tmp0_h1_4.default_stride ? NULL : &q_rope_tmp0_h1_4.stride), (q_rope_src_h1_8.default_stride ? NULL : &q_rope_src_h1_8.stride), (cos_shared_8.default_stride ? NULL : &cos_shared_8.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_4.addr, q_rope_src_h1_8.addr, sin_shared_8.addr, &q_rope_tmp1_h1_4.shape, (q_rope_tmp1_h1_4.default_stride ? NULL : &q_rope_tmp1_h1_4.stride), (q_rope_src_h1_8.default_stride ? NULL : &q_rope_src_h1_8.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_4_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_4_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_4_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_4.addr, q_rope_src_h1_8.addr, q_rope_tmp2_h1_4_scalar_DT_BFP16, &q_rope_tmp2_h1_4.shape, (q_rope_tmp2_h1_4.default_stride ? NULL : &q_rope_tmp2_h1_4.stride), (q_rope_src_h1_8.default_stride ? NULL : &q_rope_src_h1_8.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_4.addr, q_rope_tmp2_h1_4.addr, sin_shared_8.addr, &q_rope_tmp3_h1_4.shape, (q_rope_tmp3_h1_4.default_stride ? NULL : &q_rope_tmp3_h1_4.stride), (q_rope_tmp2_h1_4.default_stride ? NULL : &q_rope_tmp2_h1_4.stride), (sin_shared_8.default_stride ? NULL : &sin_shared_8.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_8.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_8.shape.n, .c = q_rope_h1_8.shape.c, .h = q_rope_h1_8.shape.h, .w = q_rope_h1_8.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_8.addr, q_rope_tmp0_h1_4.addr, q_rope_tmp3_h1_4.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_8.addr + 2, q_rope_tmp0_h1_4.addr + 2, q_rope_tmp1_h1_4.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_24 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_24.stride, 0, &q_nope_head_24.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_24 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_24.stride, 0, &q_wukv_acc_24.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_24 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_24.stride, 0, &q_wukv_head_24.shape, DT_BFP16);
    __ppl_tensor_info score_tile_16 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_16.stride, 0, &score_tile_16.shape, DT_FP32);
    __ppl_tensor_info prob_tile_16 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_16.stride, 0, &prob_tile_16.shape, DT_BFP16);
    __ppl_tensor_info scores_max_8 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_8.stride, 0, &scores_max_8.shape, DT_FP32);
    __ppl_tensor_info scores_sum_8 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_8.stride, 0, &scores_sum_8.shape, DT_FP32);
    __ppl_tensor_info work_score0_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_8.stride, 0, &work_score0_8.shape, DT_FP32);
    __ppl_tensor_info work_score1_8 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_8.stride, 0, &work_score1_8.shape, DT_FP32);
    __ppl_tensor_info coeff_8 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_8.stride, 0, &coeff_8.shape, DT_FP32);
    __ppl_tensor_info table_8 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_8.stride, 0, &table_8.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_16 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_16.stride, 0, &ctx_acc_16.shape, DT_FP32);
    __ppl_tensor_info ctx_head_24 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_24.stride, 0, &ctx_head_24.shape, DT_BFP16);
    __ppl_tensor_info out_part_24 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_24.stride, 0, &out_part_24.shape, DT_FP32);
    __ppl_tensor_info out_cast_24 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_24.stride, 0, &out_cast_24.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_19 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_16.stride, .addr = q_upper_h0_16.addr + ((0) * q_upper_h0_16.stride.c+(0) * q_upper_h0_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_16.stride.c+(0) * q_upper_h0_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_16.default_stride};
    __ppl_tensor_info q_nope_head_25 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_24.stride, .addr = q_nope_head_24.addr + ((0) * q_nope_head_24.stride.c+(0) * q_nope_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_24.stride.c+(0) * q_nope_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_24.default_stride};
    tpu_bdc_cpy(q_nope_head_25.addr, q_upper_h0_19.addr, &q_nope_head_25.shape, (q_nope_head_25.default_stride ? NULL : &q_nope_head_25.stride), (q_upper_h0_19.default_stride ? NULL : &q_upper_h0_19.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_24_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_24.addr, q_wukv_acc_24_scalar_f32, &q_wukv_acc_24.shape, (q_wukv_acc_24.default_stride ? NULL : &q_wukv_acc_24.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_26 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_24.stride, .addr = q_nope_head_24.addr + (((0) / 64) * q_nope_head_24.stride.c+(0) * q_nope_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_24.stride.c+(0) * q_nope_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_26 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_24.stride, .addr = wukv_full_tile_24.addr + (((0) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_25 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_24.stride, .addr = q_wukv_acc_24.addr + (((0) / 64) * q_wukv_acc_24.stride.c+(0) * q_wukv_acc_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_24.stride.c+(0) * q_wukv_acc_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_25.addr, q_nope_head_26.addr, wukv_full_tile_26.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_26 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_24.stride, .addr = q_wukv_acc_24.addr + ((0) * q_wukv_acc_24.stride.c+(0) * q_wukv_acc_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_24.stride.c+(0) * q_wukv_acc_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_24.default_stride};
    __ppl_tensor_info q_wukv_head_25 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_24.stride, .addr = q_wukv_head_24.addr + ((0) * q_wukv_head_24.stride.c+(0) * q_wukv_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_24.stride.c+(0) * q_wukv_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_24.default_stride};
    tpu_bdc_cast(q_wukv_head_25.addr, q_wukv_acc_26.addr, &q_wukv_head_25.shape, (q_wukv_head_25.default_stride ? NULL : &q_wukv_head_25.stride), (q_wukv_acc_26.default_stride ? NULL : &q_wukv_acc_26.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_26 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_24.stride, .addr = q_wukv_head_24.addr + (((0) / 64) * q_wukv_head_24.stride.c+(0) * q_wukv_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_24.stride.c+(0) * q_wukv_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_18 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_16.stride, .addr = kv_cache_full_16.addr + (((0) / 64) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_13 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_12.stride, .addr = score_latent_h0_12.addr + (((0) / 64) * score_latent_h0_12.stride.c+(0) * score_latent_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_12.stride.c+(0) * score_latent_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_13.addr, q_wukv_head_26.addr, kv_cache_full_18.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_9 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_8.stride, .addr = q_rope_h0_8.addr + (((0) / 64) * q_rope_h0_8.stride.c+(0) * q_rope_h0_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_8.stride.c+(0) * q_rope_h0_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_18 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_16.stride, .addr = pe_score_tile_16.addr + (((0) / 64) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_13 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_12.stride, .addr = score_rope_h0_12.addr + (((0) / 64) * score_rope_h0_12.stride.c+(0) * score_rope_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_12.stride.c+(0) * score_rope_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_13.addr, q_rope_h0_9.addr, pe_score_tile_18.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_16.addr, score_latent_h0_12.addr, score_rope_h0_12.addr, &score_tile_16.shape, (score_tile_16.default_stride ? NULL : &score_tile_16.stride), (score_latent_h0_12.default_stride ? NULL : &score_latent_h0_12.stride), (score_rope_h0_12.default_stride ? NULL : &score_rope_h0_12.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_16.addr, score_tile_16.addr, mask_row_8.addr, &score_tile_16.shape, (score_tile_16.default_stride ? NULL : &score_tile_16.stride), (score_tile_16.default_stride ? NULL : &score_tile_16.stride), (mask_row_8.default_stride ? NULL : &mask_row_8.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_16.addr, score_tile_16.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_16.shape, (score_tile_16.default_stride ? NULL : &score_tile_16.stride), (score_tile_16.default_stride ? NULL : &score_tile_16.stride), DT_FP32);
    {
    scalar_t scores_max_8_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_8.addr, scores_max_8_scalar_f32, &scores_max_8.shape, (scores_max_8.default_stride ? NULL : &scores_max_8.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_8 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_8.stride, 0, &tmp_buffer_max_8.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_16.shape.n, score_tile_16.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_16.shape.n, score_tile_16.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_16.shape.n, score_tile_16.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_16.shape.n, score_tile_16.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_16.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, score_tile_16.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_16.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_16.shape.w) {
          dim4 fill_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, align_w - score_tile_16.shape.w};
          int elem_size = 4;
          int offset = score_tile_16.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_16.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_16.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_16.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_8_stride;
    tpu_aligned_stride(&scores_max_8_stride, 0, &scores_max_8.shape, DT_FP32);
    scores_max_8_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_16.addr, score_tile_16.addr, scores_max_8.addr, &score_tile_16.shape, (score_tile_16.default_stride ? NULL : &score_tile_16.stride), (score_tile_16.default_stride ? NULL : &score_tile_16.stride), &scores_max_8_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_8.addr);
    tpu_bdc_load_fp32_exp_table(table_8.addr);
    tpu_bdc_fp32_exp(score_tile_16.addr, score_tile_16.addr, work_score0_8.addr, work_score1_8.addr, coeff_8.addr, table_8.addr, &score_tile_16.shape);
    __ppl_tensor_info tmp_buffer_sum_8 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_8.stride, 0, &tmp_buffer_sum_8.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_16.shape.n, score_tile_16.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_16.shape.n, score_tile_16.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_16.shape.n, score_tile_16.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_16.shape.n, score_tile_16.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_16.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, score_tile_16.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_16.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_16.shape.w) {
          dim4 fill_shape = {score_tile_16.shape.n, score_tile_16.shape.c, 1, align_w - score_tile_16.shape.w};
          int elem_size = 4;
          int offset = score_tile_16.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_16.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_16.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_16.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_8.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_17 = {.shape = {1, 1, 1, 192} , .stride = score_tile_16.stride, .addr = score_tile_16.addr + ((0) * score_tile_16.stride.c+(0) * score_tile_16.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_16.stride.c+(0) * score_tile_16.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_16.default_stride};
    __ppl_tensor_info prob_tile_17 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_16.stride, .addr = prob_tile_16.addr + ((0) * prob_tile_16.stride.c+(0) * prob_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_16.stride.c+(0) * prob_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_16.default_stride};
    tpu_bdc_cast(prob_tile_17.addr, score_tile_17.addr, &prob_tile_17.shape, (prob_tile_17.default_stride ? NULL : &prob_tile_17.stride), (score_tile_17.default_stride ? NULL : &score_tile_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_16_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_16.addr, ctx_acc_16_scalar_f32, &ctx_acc_16.shape, (ctx_acc_16.default_stride ? NULL : &ctx_acc_16.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_16.addr, prob_tile_16.addr, kv_cache_full_16.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_8_stride;
    tpu_aligned_stride(&scores_sum_8_stride, 0, &scores_sum_8.shape, DT_FP32);
    scores_sum_8_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_16.addr, ctx_acc_16.addr, scores_sum_8.addr, &ctx_acc_16.shape, (ctx_acc_16.default_stride ? NULL : &ctx_acc_16.stride), (ctx_acc_16.default_stride ? NULL : &ctx_acc_16.stride), &scores_sum_8_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_17 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_16.stride, .addr = ctx_acc_16.addr + ((0) * ctx_acc_16.stride.c+(0) * ctx_acc_16.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_16.stride.c+(0) * ctx_acc_16.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_16.default_stride};
    __ppl_tensor_info ctx_head_25 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_24.stride, .addr = ctx_head_24.addr + ((0) * ctx_head_24.stride.c+(0) * ctx_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_24.stride.c+(0) * ctx_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_24.default_stride};
    tpu_bdc_cast(ctx_head_25.addr, ctx_acc_17.addr, &ctx_head_25.shape, (ctx_head_25.default_stride ? NULL : &ctx_head_25.stride), (ctx_acc_17.default_stride ? NULL : &ctx_acc_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_24_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_24.addr, out_part_24_scalar_f32, &out_part_24.shape, (out_part_24.default_stride ? NULL : &out_part_24.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_26 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_24.stride, .addr = ctx_head_24.addr + (((0) / 64) * ctx_head_24.stride.c+(0) * ctx_head_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_24.stride.c+(0) * ctx_head_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_27 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_24.stride, .addr = wukv_full_tile_24.addr + (((128) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_25 = {.shape = {1, 1, 1, 128}, .stride = out_part_24.stride, .addr = out_part_24.addr + (((0) / 64) * out_part_24.stride.c+(0) * out_part_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_24.stride.c+(0) * out_part_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_25.addr, ctx_head_26.addr, wukv_full_tile_27.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_26 = {.shape = {1, 1, 1, 128} , .stride = out_part_24.stride, .addr = out_part_24.addr + ((0) * out_part_24.stride.c+(0) * out_part_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_24.stride.c+(0) * out_part_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_24.default_stride};
    __ppl_tensor_info out_cast_25 = {.shape = {1, 1, 1, 128} , .stride = out_cast_24.stride, .addr = out_cast_24.addr + ((0) * out_cast_24.stride.c+(0) * out_cast_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_24.stride.c+(0) * out_cast_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_24.default_stride};
    tpu_bdc_cast(out_cast_25.addr, out_part_26.addr, &out_cast_25.shape, (out_cast_25.default_stride ? NULL : &out_cast_25.stride), (out_part_26.default_stride ? NULL : &out_part_26.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_26 = {.shape = {1, 1, 1, 128} , .stride = out_cast_24.stride, .addr = out_cast_24.addr + ((0) * out_cast_24.stride.c+(0) * out_cast_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_24.stride.c+(0) * out_cast_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_24.default_stride};
    __ppl_tensor_info OUT_8 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1024) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1024) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_8.addr, out_cast_26.addr, &OUT_8.shape, (OUT_8.default_stride ? NULL : &OUT_8.stride), (out_cast_26.default_stride ? NULL : &out_cast_26.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_27 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_27.stride, 0, &q_nope_head_27.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_27 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_27.stride, 0, &q_wukv_acc_27.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_27 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_27.stride, 0, &q_wukv_head_27.shape, DT_BFP16);
    __ppl_tensor_info score_tile_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_18.stride, 0, &score_tile_18.shape, DT_FP32);
    __ppl_tensor_info prob_tile_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_18.stride, 0, &prob_tile_18.shape, DT_BFP16);
    __ppl_tensor_info scores_max_9 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_9.stride, 0, &scores_max_9.shape, DT_FP32);
    __ppl_tensor_info scores_sum_9 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_9.stride, 0, &scores_sum_9.shape, DT_FP32);
    __ppl_tensor_info work_score0_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_9.stride, 0, &work_score0_9.shape, DT_FP32);
    __ppl_tensor_info work_score1_9 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_9.stride, 0, &work_score1_9.shape, DT_FP32);
    __ppl_tensor_info coeff_9 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_9.stride, 0, &coeff_9.shape, DT_FP32);
    __ppl_tensor_info table_9 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_9.stride, 0, &table_9.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_18 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_18.stride, 0, &ctx_acc_18.shape, DT_FP32);
    __ppl_tensor_info ctx_head_27 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_27.stride, 0, &ctx_head_27.shape, DT_BFP16);
    __ppl_tensor_info out_part_27 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_27.stride, 0, &out_part_27.shape, DT_FP32);
    __ppl_tensor_info out_cast_27 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_27.stride, 0, &out_cast_27.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_19 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_16.stride, .addr = q_upper_h1_16.addr + ((0) * q_upper_h1_16.stride.c+(0) * q_upper_h1_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_16.stride.c+(0) * q_upper_h1_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_16.default_stride};
    __ppl_tensor_info q_nope_head_28 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_27.stride, .addr = q_nope_head_27.addr + ((0) * q_nope_head_27.stride.c+(0) * q_nope_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_27.stride.c+(0) * q_nope_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_27.default_stride};
    tpu_bdc_cpy(q_nope_head_28.addr, q_upper_h1_19.addr, &q_nope_head_28.shape, (q_nope_head_28.default_stride ? NULL : &q_nope_head_28.stride), (q_upper_h1_19.default_stride ? NULL : &q_upper_h1_19.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_27_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_27.addr, q_wukv_acc_27_scalar_f32, &q_wukv_acc_27.shape, (q_wukv_acc_27.default_stride ? NULL : &q_wukv_acc_27.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_29 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_27.stride, .addr = q_nope_head_27.addr + (((0) / 64) * q_nope_head_27.stride.c+(0) * q_nope_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_27.stride.c+(0) * q_nope_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_28 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_24.stride, .addr = wukv_full_tile_24.addr + (((256) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_28 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_27.stride, .addr = q_wukv_acc_27.addr + (((0) / 64) * q_wukv_acc_27.stride.c+(0) * q_wukv_acc_27.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_27.stride.c+(0) * q_wukv_acc_27.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_28.addr, q_nope_head_29.addr, wukv_full_tile_28.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_29 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_27.stride, .addr = q_wukv_acc_27.addr + ((0) * q_wukv_acc_27.stride.c+(0) * q_wukv_acc_27.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_27.stride.c+(0) * q_wukv_acc_27.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_27.default_stride};
    __ppl_tensor_info q_wukv_head_28 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_27.stride, .addr = q_wukv_head_27.addr + ((0) * q_wukv_head_27.stride.c+(0) * q_wukv_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_27.stride.c+(0) * q_wukv_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_27.default_stride};
    tpu_bdc_cast(q_wukv_head_28.addr, q_wukv_acc_29.addr, &q_wukv_head_28.shape, (q_wukv_head_28.default_stride ? NULL : &q_wukv_head_28.stride), (q_wukv_acc_29.default_stride ? NULL : &q_wukv_acc_29.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_29 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_27.stride, .addr = q_wukv_head_27.addr + (((0) / 64) * q_wukv_head_27.stride.c+(0) * q_wukv_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_27.stride.c+(0) * q_wukv_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_19 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_16.stride, .addr = kv_cache_full_16.addr + (((0) / 64) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_16.stride.c+(0) * kv_cache_full_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_14 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_12.stride, .addr = score_latent_h0_12.addr + (((0) / 64) * score_latent_h0_12.stride.c+(0) * score_latent_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_12.stride.c+(0) * score_latent_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_14.addr, q_wukv_head_29.addr, kv_cache_full_19.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_9 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_8.stride, .addr = q_rope_h1_8.addr + (((0) / 64) * q_rope_h1_8.stride.c+(0) * q_rope_h1_8.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_8.stride.c+(0) * q_rope_h1_8.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_19 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_16.stride, .addr = pe_score_tile_16.addr + (((0) / 64) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_16.stride.c+(0) * pe_score_tile_16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_14 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_12.stride, .addr = score_rope_h0_12.addr + (((0) / 64) * score_rope_h0_12.stride.c+(0) * score_rope_h0_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_12.stride.c+(0) * score_rope_h0_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_14.addr, q_rope_h1_9.addr, pe_score_tile_19.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_18.addr, score_latent_h0_12.addr, score_rope_h0_12.addr, &score_tile_18.shape, (score_tile_18.default_stride ? NULL : &score_tile_18.stride), (score_latent_h0_12.default_stride ? NULL : &score_latent_h0_12.stride), (score_rope_h0_12.default_stride ? NULL : &score_rope_h0_12.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_18.addr, score_tile_18.addr, mask_row_8.addr, &score_tile_18.shape, (score_tile_18.default_stride ? NULL : &score_tile_18.stride), (score_tile_18.default_stride ? NULL : &score_tile_18.stride), (mask_row_8.default_stride ? NULL : &mask_row_8.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_18.addr, score_tile_18.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_18.shape, (score_tile_18.default_stride ? NULL : &score_tile_18.stride), (score_tile_18.default_stride ? NULL : &score_tile_18.stride), DT_FP32);
    {
    scalar_t scores_max_9_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_9.addr, scores_max_9_scalar_f32, &scores_max_9.shape, (scores_max_9.default_stride ? NULL : &scores_max_9.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_9 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_9.stride, 0, &tmp_buffer_max_9.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_18.shape.n, score_tile_18.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_18.shape.n, score_tile_18.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_18.shape.n, score_tile_18.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_18.shape.n, score_tile_18.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_18.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, score_tile_18.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_18.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_18.shape.w) {
          dim4 fill_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, align_w - score_tile_18.shape.w};
          int elem_size = 4;
          int offset = score_tile_18.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_18.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_18.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_18.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_9_stride;
    tpu_aligned_stride(&scores_max_9_stride, 0, &scores_max_9.shape, DT_FP32);
    scores_max_9_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_18.addr, score_tile_18.addr, scores_max_9.addr, &score_tile_18.shape, (score_tile_18.default_stride ? NULL : &score_tile_18.stride), (score_tile_18.default_stride ? NULL : &score_tile_18.stride), &scores_max_9_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_9.addr);
    tpu_bdc_load_fp32_exp_table(table_9.addr);
    tpu_bdc_fp32_exp(score_tile_18.addr, score_tile_18.addr, work_score0_9.addr, work_score1_9.addr, coeff_9.addr, table_9.addr, &score_tile_18.shape);
    __ppl_tensor_info tmp_buffer_sum_9 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_9.stride, 0, &tmp_buffer_sum_9.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_18.shape.n, score_tile_18.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_18.shape.n, score_tile_18.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_18.shape.n, score_tile_18.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_18.shape.n, score_tile_18.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_18.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, score_tile_18.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_18.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_18.shape.w) {
          dim4 fill_shape = {score_tile_18.shape.n, score_tile_18.shape.c, 1, align_w - score_tile_18.shape.w};
          int elem_size = 4;
          int offset = score_tile_18.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_18.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_18.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_18.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_9.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_19 = {.shape = {1, 1, 1, 192} , .stride = score_tile_18.stride, .addr = score_tile_18.addr + ((0) * score_tile_18.stride.c+(0) * score_tile_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_18.stride.c+(0) * score_tile_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_18.default_stride};
    __ppl_tensor_info prob_tile_19 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_18.stride, .addr = prob_tile_18.addr + ((0) * prob_tile_18.stride.c+(0) * prob_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_18.stride.c+(0) * prob_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_18.default_stride};
    tpu_bdc_cast(prob_tile_19.addr, score_tile_19.addr, &prob_tile_19.shape, (prob_tile_19.default_stride ? NULL : &prob_tile_19.stride), (score_tile_19.default_stride ? NULL : &score_tile_19.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_18.addr, ctx_acc_18_scalar_f32, &ctx_acc_18.shape, (ctx_acc_18.default_stride ? NULL : &ctx_acc_18.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_18.addr, prob_tile_18.addr, kv_cache_full_16.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_9_stride;
    tpu_aligned_stride(&scores_sum_9_stride, 0, &scores_sum_9.shape, DT_FP32);
    scores_sum_9_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_18.addr, ctx_acc_18.addr, scores_sum_9.addr, &ctx_acc_18.shape, (ctx_acc_18.default_stride ? NULL : &ctx_acc_18.stride), (ctx_acc_18.default_stride ? NULL : &ctx_acc_18.stride), &scores_sum_9_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_19 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_18.stride, .addr = ctx_acc_18.addr + ((0) * ctx_acc_18.stride.c+(0) * ctx_acc_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_18.stride.c+(0) * ctx_acc_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_18.default_stride};
    __ppl_tensor_info ctx_head_28 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_27.stride, .addr = ctx_head_27.addr + ((0) * ctx_head_27.stride.c+(0) * ctx_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_27.stride.c+(0) * ctx_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_27.default_stride};
    tpu_bdc_cast(ctx_head_28.addr, ctx_acc_19.addr, &ctx_head_28.shape, (ctx_head_28.default_stride ? NULL : &ctx_head_28.stride), (ctx_acc_19.default_stride ? NULL : &ctx_acc_19.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_27_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_27.addr, out_part_27_scalar_f32, &out_part_27.shape, (out_part_27.default_stride ? NULL : &out_part_27.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_29 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_27.stride, .addr = ctx_head_27.addr + (((0) / 64) * ctx_head_27.stride.c+(0) * ctx_head_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_27.stride.c+(0) * ctx_head_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_29 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_24.stride, .addr = wukv_full_tile_24.addr + (((384) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_24.stride.c+(0) * wukv_full_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_28 = {.shape = {1, 1, 1, 128}, .stride = out_part_27.stride, .addr = out_part_27.addr + (((0) / 64) * out_part_27.stride.c+(0) * out_part_27.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_27.stride.c+(0) * out_part_27.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_28.addr, ctx_head_29.addr, wukv_full_tile_29.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_29 = {.shape = {1, 1, 1, 128} , .stride = out_part_27.stride, .addr = out_part_27.addr + ((0) * out_part_27.stride.c+(0) * out_part_27.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_27.stride.c+(0) * out_part_27.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_27.default_stride};
    __ppl_tensor_info out_cast_28 = {.shape = {1, 1, 1, 128} , .stride = out_cast_27.stride, .addr = out_cast_27.addr + ((0) * out_cast_27.stride.c+(0) * out_cast_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_27.stride.c+(0) * out_cast_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_27.default_stride};
    tpu_bdc_cast(out_cast_28.addr, out_part_29.addr, &out_cast_28.shape, (out_cast_28.default_stride ? NULL : &out_cast_28.stride), (out_part_29.default_stride ? NULL : &out_part_29.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_29 = {.shape = {1, 1, 1, 128} , .stride = out_cast_27.stride, .addr = out_cast_27.addr + ((0) * out_cast_27.stride.c+(0) * out_cast_27.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_27.stride.c+(0) * out_cast_27.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_27.default_stride};
    __ppl_tensor_info OUT_9 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1152) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1152) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_9.addr, out_cast_29.addr, &OUT_9.shape, (OUT_9.default_stride ? NULL : &OUT_9.stride), (out_cast_29.default_stride ? NULL : &out_cast_29.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_15.stride, 0, &q_upper_acc_h0_15.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_20 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_20.stride, 0, &q_upper_h0_20.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_15.stride, 0, &q_upper_acc_h1_15.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_20 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_20.stride, 0, &q_upper_h1_20.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_10.stride, 0, &q_rope_src_h0_10.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_10.stride, 0, &q_rope_h0_10.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_5.stride, 0, &q_rope_tmp0_h0_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_5.stride, 0, &q_rope_tmp1_h0_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_5.stride, 0, &q_rope_tmp2_h0_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_5.stride, 0, &q_rope_tmp3_h0_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_10.stride, 0, &q_rope_src_h1_10.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_10.stride, 0, &q_rope_h1_10.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_5.stride, 0, &q_rope_tmp0_h1_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_5.stride, 0, &q_rope_tmp1_h1_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_5.stride, 0, &q_rope_tmp2_h1_5.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_5.stride, 0, &q_rope_tmp3_h1_5.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_10.stride, 0, &cos_shared_10.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_10.stride, 0, &sin_shared_10.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_10.stride, 0, &pe_cur_10.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_5.stride, 0, &pe_cur_cos_5.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_5.stride, 0, &pe_cur_sin_5.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_5.stride, 0, &pe_cur_neg_5.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_5 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_5.stride, 0, &pe_cur_neg_sin_5.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_10 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_10.stride, 0, &pe_cur_rope_10.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_30 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_30.stride, 0, &wukv_full_tile_30.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_20 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_20.stride, 0, &kv_cache_full_20.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_20 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_20.stride, 0, &pe_score_tile_20.shape, DT_BFP16);
  __ppl_tensor_info mask_row_10 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_10.stride, 0, &mask_row_10.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_15.stride, 0, &score_latent_h0_15.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_15.stride, 0, &score_rope_h0_15.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_5 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_5.stride, 0, &score_latent_h1_5.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_5 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_5.stride, 0, &score_rope_h1_5.shape, DT_FP32);
  if (tpu_workitem_index() == 5) {
    __ppl_tensor_info COS_5 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_11 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_10.stride, .addr = cos_shared_10.addr + ((0) * cos_shared_10.stride.c+(0) * cos_shared_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_10.stride.c+(0) * cos_shared_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_10.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_11.addr, COS_5.addr, &cos_shared_11.shape, (cos_shared_11.default_stride ? NULL : &cos_shared_11.stride), (COS_5.default_stride ? NULL : &COS_5.stride), DT_BFP16);
    __ppl_tensor_info SIN_5 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_11 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_10.stride, .addr = sin_shared_10.addr + ((0) * sin_shared_10.stride.c+(0) * sin_shared_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_10.stride.c+(0) * sin_shared_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_10.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_11.addr, SIN_5.addr, &sin_shared_11.shape, (sin_shared_11.default_stride ? NULL : &sin_shared_11.stride), (SIN_5.default_stride ? NULL : &SIN_5.stride), DT_BFP16);
    for (int d_10 = 0; d_10 < 4; ++d_10) {
      __ppl_tensor_info kv_cur_tile_15 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_15.stride, 0, &kv_cur_tile_15.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_5 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_10 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_10 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_16 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_15.stride, .addr = kv_cur_tile_15.addr + ((0) * kv_cur_tile_15.stride.c+(0) * kv_cur_tile_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_15.stride.c+(0) * kv_cur_tile_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_15.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_16.addr, KV_CUR_5.addr, &kv_cur_tile_16.shape, (kv_cur_tile_16.default_stride ? NULL : &kv_cur_tile_16.stride), (KV_CUR_5.default_stride ? NULL : &KV_CUR_5.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_17 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_15.stride, .addr = kv_cur_tile_15.addr + ((0) * kv_cur_tile_15.stride.c+(0) * kv_cur_tile_15.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_15.stride.c+(0) * kv_cur_tile_15.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_15.default_stride};
      __ppl_tensor_info KVcache_OUT_10 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_10 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_10 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_10.addr, kv_cur_tile_17.addr, &KVcache_OUT_10.shape, (KVcache_OUT_10.default_stride ? NULL : &KVcache_OUT_10.stride), (kv_cur_tile_17.default_stride ? NULL : &kv_cur_tile_17.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_5 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_11 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_10.stride, .addr = pe_cur_10.addr + ((0) * pe_cur_10.stride.c+(0) * pe_cur_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_10.stride.c+(0) * pe_cur_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_10.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_11.addr, PE_CUR_5.addr, &pe_cur_11.shape, (pe_cur_11.default_stride ? NULL : &pe_cur_11.stride), (PE_CUR_5.default_stride ? NULL : &PE_CUR_5.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_5.addr, pe_cur_10.addr, cos_shared_10.addr, &pe_cur_cos_5.shape, (pe_cur_cos_5.default_stride ? NULL : &pe_cur_cos_5.stride), (pe_cur_10.default_stride ? NULL : &pe_cur_10.stride), (cos_shared_10.default_stride ? NULL : &cos_shared_10.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_5.addr, pe_cur_10.addr, sin_shared_10.addr, &pe_cur_sin_5.shape, (pe_cur_sin_5.default_stride ? NULL : &pe_cur_sin_5.stride), (pe_cur_10.default_stride ? NULL : &pe_cur_10.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_5_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_5_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_5_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_5.addr, pe_cur_10.addr, pe_cur_neg_5_scalar_DT_BFP16, &pe_cur_neg_5.shape, (pe_cur_neg_5.default_stride ? NULL : &pe_cur_neg_5.stride), (pe_cur_10.default_stride ? NULL : &pe_cur_10.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_5.addr, pe_cur_neg_5.addr, sin_shared_10.addr, &pe_cur_neg_sin_5.shape, (pe_cur_neg_sin_5.default_stride ? NULL : &pe_cur_neg_sin_5.stride), (pe_cur_neg_5.default_stride ? NULL : &pe_cur_neg_5.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_10.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_10.shape.n, .c = pe_cur_rope_10.shape.c, .h = pe_cur_rope_10.shape.h, .w = pe_cur_rope_10.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_10.addr, pe_cur_cos_5.addr, pe_cur_neg_sin_5.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_10.addr + 2, pe_cur_cos_5.addr + 2, pe_cur_sin_5.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_11 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_10.stride, .addr = pe_cur_rope_10.addr + ((0) * pe_cur_rope_10.stride.c+(0) * pe_cur_rope_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_10.stride.c+(0) * pe_cur_rope_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_10.default_stride};
    __ppl_tensor_info PEcache_OUT_10 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_10.addr, pe_cur_rope_11.addr, &PEcache_OUT_10.shape, (PEcache_OUT_10.default_stride ? NULL : &PEcache_OUT_10.stride), (pe_cur_rope_11.default_stride ? NULL : &pe_cur_rope_11.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_15.addr, q_upper_acc_h0_15_scalar_f32, &q_upper_acc_h0_15.shape, (q_upper_acc_h0_15.default_stride ? NULL : &q_upper_acc_h0_15.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_15.addr, q_upper_acc_h1_15_scalar_f32, &q_upper_acc_h1_15.shape, (q_upper_acc_h1_15.default_stride ? NULL : &q_upper_acc_h1_15.stride), DT_FP32);
    }
    for (int k_5 = 0; k_5 < 12; ++k_5) {
      __ppl_tensor_info q_tile_20 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_20.stride, 0, &q_tile_20.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_15 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_15.stride, 0, &wuq_fp8_15.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_40 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_40.stride, 0, &wuq_scale_stage_40.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_20 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_20.stride, 0, &wuq_tile_20.shape, DT_BFP16);
      __ppl_tensor_info Q_5 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_5 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_5 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_21 = {.shape = {1, 1, 1, 128} , .stride = q_tile_20.stride, .addr = q_tile_20.addr + ((0) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_20.default_stride};
      tpu_gdma_cpy_S2L(q_tile_21.addr, Q_5.addr, &q_tile_21.shape, (q_tile_21.default_stride ? NULL : &q_tile_21.stride), (Q_5.default_stride ? NULL : &Q_5.stride), DT_BFP16);
      __ppl_tensor_info WUQ_5 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((1920) * 1536+((k_5 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((1920) * 1536+((k_5 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_16 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_15.stride, .addr = wuq_fp8_15.addr + ((0) * wuq_fp8_15.stride.c+(0) * wuq_fp8_15.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_15.stride.c+(0) * wuq_fp8_15.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_15.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_16.addr, WUQ_5.addr, &wuq_fp8_16.shape, (wuq_fp8_16.default_stride ? NULL : &wuq_fp8_16.stride), (WUQ_5.default_stride ? NULL : &WUQ_5.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_90 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_90.stride, 0, &wuq_scale_scalar_90.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_30 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((15) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((15) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_91 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_90.stride, .addr = wuq_scale_scalar_90.addr + ((0) * wuq_scale_scalar_90.stride.c+(0) * wuq_scale_scalar_90.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_90.stride.c+(0) * wuq_scale_scalar_90.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_90.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_91.addr, WUQ_SCALE_30.addr, &wuq_scale_scalar_91.shape, (wuq_scale_scalar_91.default_stride ? NULL : &wuq_scale_scalar_91.stride), (WUQ_SCALE_30.default_stride ? NULL : &WUQ_SCALE_30.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_92 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_90.stride, .addr = wuq_scale_scalar_90.addr + (((0) / 64) * wuq_scale_scalar_90.stride.c+(0) * wuq_scale_scalar_90.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_90.stride.c+(0) * wuq_scale_scalar_90.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_41 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((0) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_41.addr, wuq_scale_scalar_92.addr, &wuq_scale_stage_41.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_93 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_93.stride, 0, &wuq_scale_scalar_93.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_31 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((15) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((15) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_94 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_93.stride, .addr = wuq_scale_scalar_93.addr + ((0) * wuq_scale_scalar_93.stride.c+(0) * wuq_scale_scalar_93.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_93.stride.c+(0) * wuq_scale_scalar_93.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_93.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_94.addr, WUQ_SCALE_31.addr, &wuq_scale_scalar_94.shape, (wuq_scale_scalar_94.default_stride ? NULL : &wuq_scale_scalar_94.stride), (WUQ_SCALE_31.default_stride ? NULL : &WUQ_SCALE_31.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_95 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_93.stride, .addr = wuq_scale_scalar_93.addr + (((0) / 64) * wuq_scale_scalar_93.stride.c+(0) * wuq_scale_scalar_93.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_93.stride.c+(0) * wuq_scale_scalar_93.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_42 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((64) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_42.addr, wuq_scale_scalar_95.addr, &wuq_scale_stage_42.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_96 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_96.stride, 0, &wuq_scale_scalar_96.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_32 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((16) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((16) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_97 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_96.stride, .addr = wuq_scale_scalar_96.addr + ((0) * wuq_scale_scalar_96.stride.c+(0) * wuq_scale_scalar_96.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_96.stride.c+(0) * wuq_scale_scalar_96.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_96.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_97.addr, WUQ_SCALE_32.addr, &wuq_scale_scalar_97.shape, (wuq_scale_scalar_97.default_stride ? NULL : &wuq_scale_scalar_97.stride), (WUQ_SCALE_32.default_stride ? NULL : &WUQ_SCALE_32.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_98 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_96.stride, .addr = wuq_scale_scalar_96.addr + (((0) / 64) * wuq_scale_scalar_96.stride.c+(0) * wuq_scale_scalar_96.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_96.stride.c+(0) * wuq_scale_scalar_96.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_43 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((128) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_43.addr, wuq_scale_scalar_98.addr, &wuq_scale_stage_43.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_99 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_99.stride, 0, &wuq_scale_scalar_99.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_33 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((16) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((16) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_100 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_99.stride, .addr = wuq_scale_scalar_99.addr + ((0) * wuq_scale_scalar_99.stride.c+(0) * wuq_scale_scalar_99.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_99.stride.c+(0) * wuq_scale_scalar_99.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_99.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_100.addr, WUQ_SCALE_33.addr, &wuq_scale_scalar_100.shape, (wuq_scale_scalar_100.default_stride ? NULL : &wuq_scale_scalar_100.stride), (WUQ_SCALE_33.default_stride ? NULL : &WUQ_SCALE_33.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_101 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_99.stride, .addr = wuq_scale_scalar_99.addr + (((0) / 64) * wuq_scale_scalar_99.stride.c+(0) * wuq_scale_scalar_99.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_99.stride.c+(0) * wuq_scale_scalar_99.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_44 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((192) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_44.addr, wuq_scale_scalar_101.addr, &wuq_scale_stage_44.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_102 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_102.stride, 0, &wuq_scale_scalar_102.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_34 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((17) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((17) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_103 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_102.stride, .addr = wuq_scale_scalar_102.addr + ((0) * wuq_scale_scalar_102.stride.c+(0) * wuq_scale_scalar_102.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_102.stride.c+(0) * wuq_scale_scalar_102.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_102.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_103.addr, WUQ_SCALE_34.addr, &wuq_scale_scalar_103.shape, (wuq_scale_scalar_103.default_stride ? NULL : &wuq_scale_scalar_103.stride), (WUQ_SCALE_34.default_stride ? NULL : &WUQ_SCALE_34.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_104 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_102.stride, .addr = wuq_scale_scalar_102.addr + (((0) / 64) * wuq_scale_scalar_102.stride.c+(0) * wuq_scale_scalar_102.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_102.stride.c+(0) * wuq_scale_scalar_102.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_45 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((256) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_45.addr, wuq_scale_scalar_104.addr, &wuq_scale_stage_45.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_105 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_105.stride, 0, &wuq_scale_scalar_105.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_35 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((17) * 12+(k_5) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((17) * 12+(k_5) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_106 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_105.stride, .addr = wuq_scale_scalar_105.addr + ((0) * wuq_scale_scalar_105.stride.c+(0) * wuq_scale_scalar_105.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_105.stride.c+(0) * wuq_scale_scalar_105.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_105.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_106.addr, WUQ_SCALE_35.addr, &wuq_scale_scalar_106.shape, (wuq_scale_scalar_106.default_stride ? NULL : &wuq_scale_scalar_106.stride), (WUQ_SCALE_35.default_stride ? NULL : &WUQ_SCALE_35.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_107 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_105.stride, .addr = wuq_scale_scalar_105.addr + (((0) / 64) * wuq_scale_scalar_105.stride.c+(0) * wuq_scale_scalar_105.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_105.stride.c+(0) * wuq_scale_scalar_105.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_46 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((320) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_46.addr, wuq_scale_scalar_107.addr, &wuq_scale_stage_46.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_17 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_15.stride, .addr = wuq_fp8_15.addr + (((0) / 64) * wuq_fp8_15.stride.c+(0) * wuq_fp8_15.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_15.stride.c+(0) * wuq_fp8_15.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_47 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_40.stride, .addr = wuq_scale_stage_40.addr + (((0) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_40.stride.c+(0) * wuq_scale_stage_40.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_21 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_20.stride, .addr = wuq_tile_20.addr + (((0) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_21.addr, wuq_fp8_17.addr, &wuq_tile_21.shape, (wuq_tile_21.default_stride ? NULL : &wuq_tile_21.stride), (wuq_fp8_17.default_stride ? NULL : &wuq_fp8_17.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_47_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_47_block_stride, 0, &wuq_scale_stage_47.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_47_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_21.addr, wuq_tile_21.addr, wuq_scale_stage_47.addr, &wuq_tile_21.shape, (wuq_tile_21.default_stride ? NULL : &wuq_tile_21.stride), (wuq_tile_21.default_stride ? NULL : &wuq_tile_21.stride), &wuq_scale_stage_47_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_22 = {.shape = {1, 1, 1, 128}, .stride = q_tile_20.stride, .addr = q_tile_20.addr + (((0) / 64) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_22 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_20.stride, .addr = wuq_tile_20.addr + (((0) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_16 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_15.stride, .addr = q_upper_acc_h0_15.addr + (((0) / 64) * q_upper_acc_h0_15.stride.c+(0) * q_upper_acc_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_15.stride.c+(0) * q_upper_acc_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_16.addr, q_tile_22.addr, wuq_tile_22.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_23 = {.shape = {1, 1, 1, 128}, .stride = q_tile_20.stride, .addr = q_tile_20.addr + (((0) / 64) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_20.stride.c+(0) * q_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_23 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_20.stride, .addr = wuq_tile_20.addr + (((192) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_20.stride.c+(0) * wuq_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_16 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_15.stride, .addr = q_upper_acc_h1_15.addr + (((0) / 64) * q_upper_acc_h1_15.stride.c+(0) * q_upper_acc_h1_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_15.stride.c+(0) * q_upper_acc_h1_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_16.addr, q_tile_23.addr, wuq_tile_23.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_17 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_15.stride, .addr = q_upper_acc_h0_15.addr + ((0) * q_upper_acc_h0_15.stride.c+(0) * q_upper_acc_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_15.stride.c+(0) * q_upper_acc_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_15.default_stride};
    __ppl_tensor_info q_upper_h0_21 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_20.stride, .addr = q_upper_h0_20.addr + ((0) * q_upper_h0_20.stride.c+(0) * q_upper_h0_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_20.stride.c+(0) * q_upper_h0_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_20.default_stride};
    tpu_bdc_cast(q_upper_h0_21.addr, q_upper_acc_h0_17.addr, &q_upper_h0_21.shape, (q_upper_h0_21.default_stride ? NULL : &q_upper_h0_21.stride), (q_upper_acc_h0_17.default_stride ? NULL : &q_upper_acc_h0_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_17 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_15.stride, .addr = q_upper_acc_h1_15.addr + ((0) * q_upper_acc_h1_15.stride.c+(0) * q_upper_acc_h1_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_15.stride.c+(0) * q_upper_acc_h1_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_15.default_stride};
    __ppl_tensor_info q_upper_h1_21 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_20.stride, .addr = q_upper_h1_20.addr + ((0) * q_upper_h1_20.stride.c+(0) * q_upper_h1_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_20.stride.c+(0) * q_upper_h1_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_20.default_stride};
    tpu_bdc_cast(q_upper_h1_21.addr, q_upper_acc_h1_17.addr, &q_upper_h1_21.shape, (q_upper_h1_21.default_stride ? NULL : &q_upper_h1_21.stride), (q_upper_acc_h1_17.default_stride ? NULL : &q_upper_acc_h1_17.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_11 = 0; d_11 < 4; ++d_11) {
      __ppl_tensor_info wukv_fp8_15 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_15.stride, 0, &wukv_fp8_15.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_30 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_30.stride, 0, &wukv_scale_stage_30.shape, DT_BFP16);
      __ppl_tensor_info WUKV_5 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((2560) * 512+((d_11 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((2560) * 512+((d_11 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_16 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_15.stride, .addr = wukv_fp8_15.addr + ((0) * wukv_fp8_15.stride.c+(0) * wukv_fp8_15.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_15.stride.c+(0) * wukv_fp8_15.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_15.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_16.addr, WUKV_5.addr, &wukv_fp8_16.shape, (wukv_fp8_16.default_stride ? NULL : &wukv_fp8_16.stride), (WUKV_5.default_stride ? NULL : &WUKV_5.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_50 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_50.stride, 0, &wukv_scale_scalar_50.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_20 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((20) * 4+(d_11) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((20) * 4+(d_11) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_51 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_50.stride, .addr = wukv_scale_scalar_50.addr + ((0) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_50.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_51.addr, WUKV_SCALE_20.addr, &wukv_scale_scalar_51.shape, (wukv_scale_scalar_51.default_stride ? NULL : &wukv_scale_scalar_51.stride), (WUKV_SCALE_20.default_stride ? NULL : &WUKV_SCALE_20.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_52 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_50.stride, .addr = wukv_scale_scalar_50.addr + (((0) / 64) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_31 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_30.stride, .addr = wukv_scale_stage_30.addr + (((0) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_31_chunk = wukv_scale_stage_31;
      wukv_scale_stage_31_chunk.shape.c = 64;
      wukv_scale_stage_31_chunk.addr = wukv_scale_stage_31.addr + ((0) * wukv_scale_stage_31.stride.c) * 2;
      wukv_scale_stage_31_chunk.offset = wukv_scale_stage_31.offset + ((0) * wukv_scale_stage_31.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_31_chunk.addr, wukv_scale_scalar_52.addr, &wukv_scale_stage_31_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_31_chunk_1 = wukv_scale_stage_31;
      wukv_scale_stage_31_chunk_1.shape.c = 64;
      wukv_scale_stage_31_chunk_1.addr = wukv_scale_stage_31.addr + ((1) * wukv_scale_stage_31.stride.c) * 2;
      wukv_scale_stage_31_chunk_1.offset = wukv_scale_stage_31.offset + ((1) * wukv_scale_stage_31.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_31_chunk_1.addr, wukv_scale_scalar_52.addr, &wukv_scale_stage_31_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_21 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((21) * 4+(d_11) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((21) * 4+(d_11) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_53 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_50.stride, .addr = wukv_scale_scalar_50.addr + ((0) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_50.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_53.addr, WUKV_SCALE_21.addr, &wukv_scale_scalar_53.shape, (wukv_scale_scalar_53.default_stride ? NULL : &wukv_scale_scalar_53.stride), (WUKV_SCALE_21.default_stride ? NULL : &WUKV_SCALE_21.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_54 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_50.stride, .addr = wukv_scale_scalar_50.addr + (((0) / 64) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_50.stride.c+(0) * wukv_scale_scalar_50.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_32 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_30.stride, .addr = wukv_scale_stage_30.addr + (((128) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_32_chunk = wukv_scale_stage_32;
      wukv_scale_stage_32_chunk.shape.c = 64;
      wukv_scale_stage_32_chunk.addr = wukv_scale_stage_32.addr + ((0) * wukv_scale_stage_32.stride.c) * 2;
      wukv_scale_stage_32_chunk.offset = wukv_scale_stage_32.offset + ((0) * wukv_scale_stage_32.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_32_chunk.addr, wukv_scale_scalar_54.addr, &wukv_scale_stage_32_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_32_chunk_1 = wukv_scale_stage_32;
      wukv_scale_stage_32_chunk_1.shape.c = 64;
      wukv_scale_stage_32_chunk_1.addr = wukv_scale_stage_32.addr + ((1) * wukv_scale_stage_32.stride.c) * 2;
      wukv_scale_stage_32_chunk_1.offset = wukv_scale_stage_32.offset + ((1) * wukv_scale_stage_32.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_32_chunk_1.addr, wukv_scale_scalar_54.addr, &wukv_scale_stage_32_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_55 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_55.stride, 0, &wukv_scale_scalar_55.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_22 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((22) * 4+(d_11) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((22) * 4+(d_11) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_56 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_55.stride, .addr = wukv_scale_scalar_55.addr + ((0) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_55.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_56.addr, WUKV_SCALE_22.addr, &wukv_scale_scalar_56.shape, (wukv_scale_scalar_56.default_stride ? NULL : &wukv_scale_scalar_56.stride), (WUKV_SCALE_22.default_stride ? NULL : &WUKV_SCALE_22.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_57 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_55.stride, .addr = wukv_scale_scalar_55.addr + (((0) / 64) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_33 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_30.stride, .addr = wukv_scale_stage_30.addr + (((256) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_33_chunk = wukv_scale_stage_33;
      wukv_scale_stage_33_chunk.shape.c = 64;
      wukv_scale_stage_33_chunk.addr = wukv_scale_stage_33.addr + ((0) * wukv_scale_stage_33.stride.c) * 2;
      wukv_scale_stage_33_chunk.offset = wukv_scale_stage_33.offset + ((0) * wukv_scale_stage_33.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_33_chunk.addr, wukv_scale_scalar_57.addr, &wukv_scale_stage_33_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_33_chunk_1 = wukv_scale_stage_33;
      wukv_scale_stage_33_chunk_1.shape.c = 64;
      wukv_scale_stage_33_chunk_1.addr = wukv_scale_stage_33.addr + ((1) * wukv_scale_stage_33.stride.c) * 2;
      wukv_scale_stage_33_chunk_1.offset = wukv_scale_stage_33.offset + ((1) * wukv_scale_stage_33.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_33_chunk_1.addr, wukv_scale_scalar_57.addr, &wukv_scale_stage_33_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_23 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((23) * 4+(d_11) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((23) * 4+(d_11) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_58 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_55.stride, .addr = wukv_scale_scalar_55.addr + ((0) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_55.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_58.addr, WUKV_SCALE_23.addr, &wukv_scale_scalar_58.shape, (wukv_scale_scalar_58.default_stride ? NULL : &wukv_scale_scalar_58.stride), (WUKV_SCALE_23.default_stride ? NULL : &WUKV_SCALE_23.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_59 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_55.stride, .addr = wukv_scale_scalar_55.addr + (((0) / 64) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_55.stride.c+(0) * wukv_scale_scalar_55.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_34 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_30.stride, .addr = wukv_scale_stage_30.addr + (((384) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_34_chunk = wukv_scale_stage_34;
      wukv_scale_stage_34_chunk.shape.c = 64;
      wukv_scale_stage_34_chunk.addr = wukv_scale_stage_34.addr + ((0) * wukv_scale_stage_34.stride.c) * 2;
      wukv_scale_stage_34_chunk.offset = wukv_scale_stage_34.offset + ((0) * wukv_scale_stage_34.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_34_chunk.addr, wukv_scale_scalar_59.addr, &wukv_scale_stage_34_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_34_chunk_1 = wukv_scale_stage_34;
      wukv_scale_stage_34_chunk_1.shape.c = 64;
      wukv_scale_stage_34_chunk_1.addr = wukv_scale_stage_34.addr + ((1) * wukv_scale_stage_34.stride.c) * 2;
      wukv_scale_stage_34_chunk_1.offset = wukv_scale_stage_34.offset + ((1) * wukv_scale_stage_34.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_34_chunk_1.addr, wukv_scale_scalar_59.addr, &wukv_scale_stage_34_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_17 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_15.stride, .addr = wukv_fp8_15.addr + (((0) / 64) * wukv_fp8_15.stride.c+(0) * wukv_fp8_15.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_15.stride.c+(0) * wukv_fp8_15.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_35 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_30.stride, .addr = wukv_scale_stage_30.addr + (((0) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_30.stride.c+(0) * wukv_scale_stage_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_31 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_30.stride, .addr = wukv_full_tile_30.addr + (((0) / 64) * wukv_full_tile_30.stride.c+((d_11 * 128)) * wukv_full_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_30.stride.c+((d_11 * 128)) * wukv_full_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_31.addr, wukv_fp8_17.addr, &wukv_full_tile_31.shape, (wukv_full_tile_31.default_stride ? NULL : &wukv_full_tile_31.stride), (wukv_fp8_17.default_stride ? NULL : &wukv_fp8_17.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_35_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_35_block_stride, 0, &wukv_scale_stage_35.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_35_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_31.addr, wukv_full_tile_31.addr, wukv_scale_stage_35.addr, &wukv_full_tile_31.shape, (wukv_full_tile_31.default_stride ? NULL : &wukv_full_tile_31.stride), (wukv_full_tile_31.default_stride ? NULL : &wukv_full_tile_31.stride), &wukv_scale_stage_35_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_11 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_21 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_20.stride, .addr = kv_cache_full_20.addr + ((0) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_20.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_21.addr, KVcache_OUT_11.addr, &kv_cache_full_21.shape, (kv_cache_full_21.default_stride ? NULL : &kv_cache_full_21.stride), (KVcache_OUT_11.default_stride ? NULL : &KVcache_OUT_11.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_11 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_21 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_20.stride, .addr = pe_score_tile_20.addr + ((0) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_20.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_21.addr, PEcache_OUT_11.addr, &pe_score_tile_21.shape, (pe_score_tile_21.default_stride ? NULL : &pe_score_tile_21.stride), (PEcache_OUT_11.default_stride ? NULL : &PEcache_OUT_11.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_5 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_11 = {.shape = {1, 1, 1, 192} , .stride = mask_row_10.stride, .addr = mask_row_10.addr + ((0) * mask_row_10.stride.c+(0) * mask_row_10.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_10.stride.c+(0) * mask_row_10.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_10.default_stride};
    tpu_gdma_cpy_S2L(mask_row_11.addr, DECODE_MASK_5.addr, &mask_row_11.shape, (mask_row_11.default_stride ? NULL : &mask_row_11.stride), (DECODE_MASK_5.default_stride ? NULL : &DECODE_MASK_5.stride), DT_FP32);
    {
    scalar_t score_latent_h0_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_15.addr, score_latent_h0_15_scalar_f32, &score_latent_h0_15.shape, (score_latent_h0_15.default_stride ? NULL : &score_latent_h0_15.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_15_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_15.addr, score_rope_h0_15_scalar_f32, &score_rope_h0_15.shape, (score_rope_h0_15.default_stride ? NULL : &score_rope_h0_15.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_5_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_5.addr, score_latent_h1_5_scalar_f32, &score_latent_h1_5.shape, (score_latent_h1_5.default_stride ? NULL : &score_latent_h1_5.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_5_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_5.addr, score_rope_h1_5_scalar_f32, &score_rope_h1_5.shape, (score_rope_h1_5.default_stride ? NULL : &score_rope_h1_5.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_22 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_20.stride, .addr = q_upper_h0_20.addr + ((0) * q_upper_h0_20.stride.c+(128) * q_upper_h0_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_20.stride.c+(128) * q_upper_h0_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_20.default_stride};
    __ppl_tensor_info q_rope_src_h0_11 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_10.stride, .addr = q_rope_src_h0_10.addr + ((0) * q_rope_src_h0_10.stride.c+(0) * q_rope_src_h0_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_10.stride.c+(0) * q_rope_src_h0_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_10.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_11.addr, q_upper_h0_22.addr, &q_rope_src_h0_11.shape, (q_rope_src_h0_11.default_stride ? NULL : &q_rope_src_h0_11.stride), (q_upper_h0_22.default_stride ? NULL : &q_upper_h0_22.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_5.addr, q_rope_src_h0_10.addr, cos_shared_10.addr, &q_rope_tmp0_h0_5.shape, (q_rope_tmp0_h0_5.default_stride ? NULL : &q_rope_tmp0_h0_5.stride), (q_rope_src_h0_10.default_stride ? NULL : &q_rope_src_h0_10.stride), (cos_shared_10.default_stride ? NULL : &cos_shared_10.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_5.addr, q_rope_src_h0_10.addr, sin_shared_10.addr, &q_rope_tmp1_h0_5.shape, (q_rope_tmp1_h0_5.default_stride ? NULL : &q_rope_tmp1_h0_5.stride), (q_rope_src_h0_10.default_stride ? NULL : &q_rope_src_h0_10.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_5_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_5_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_5_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_5.addr, q_rope_src_h0_10.addr, q_rope_tmp2_h0_5_scalar_DT_BFP16, &q_rope_tmp2_h0_5.shape, (q_rope_tmp2_h0_5.default_stride ? NULL : &q_rope_tmp2_h0_5.stride), (q_rope_src_h0_10.default_stride ? NULL : &q_rope_src_h0_10.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_5.addr, q_rope_tmp2_h0_5.addr, sin_shared_10.addr, &q_rope_tmp3_h0_5.shape, (q_rope_tmp3_h0_5.default_stride ? NULL : &q_rope_tmp3_h0_5.stride), (q_rope_tmp2_h0_5.default_stride ? NULL : &q_rope_tmp2_h0_5.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_10.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_10.shape.n, .c = q_rope_h0_10.shape.c, .h = q_rope_h0_10.shape.h, .w = q_rope_h0_10.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_10.addr, q_rope_tmp0_h0_5.addr, q_rope_tmp3_h0_5.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_10.addr + 2, q_rope_tmp0_h0_5.addr + 2, q_rope_tmp1_h0_5.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_22 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_20.stride, .addr = q_upper_h1_20.addr + ((0) * q_upper_h1_20.stride.c+(128) * q_upper_h1_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_20.stride.c+(128) * q_upper_h1_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_20.default_stride};
    __ppl_tensor_info q_rope_src_h1_11 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_10.stride, .addr = q_rope_src_h1_10.addr + ((0) * q_rope_src_h1_10.stride.c+(0) * q_rope_src_h1_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_10.stride.c+(0) * q_rope_src_h1_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_10.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_11.addr, q_upper_h1_22.addr, &q_rope_src_h1_11.shape, (q_rope_src_h1_11.default_stride ? NULL : &q_rope_src_h1_11.stride), (q_upper_h1_22.default_stride ? NULL : &q_upper_h1_22.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_5.addr, q_rope_src_h1_10.addr, cos_shared_10.addr, &q_rope_tmp0_h1_5.shape, (q_rope_tmp0_h1_5.default_stride ? NULL : &q_rope_tmp0_h1_5.stride), (q_rope_src_h1_10.default_stride ? NULL : &q_rope_src_h1_10.stride), (cos_shared_10.default_stride ? NULL : &cos_shared_10.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_5.addr, q_rope_src_h1_10.addr, sin_shared_10.addr, &q_rope_tmp1_h1_5.shape, (q_rope_tmp1_h1_5.default_stride ? NULL : &q_rope_tmp1_h1_5.stride), (q_rope_src_h1_10.default_stride ? NULL : &q_rope_src_h1_10.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_5_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_5_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_5_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_5.addr, q_rope_src_h1_10.addr, q_rope_tmp2_h1_5_scalar_DT_BFP16, &q_rope_tmp2_h1_5.shape, (q_rope_tmp2_h1_5.default_stride ? NULL : &q_rope_tmp2_h1_5.stride), (q_rope_src_h1_10.default_stride ? NULL : &q_rope_src_h1_10.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_5.addr, q_rope_tmp2_h1_5.addr, sin_shared_10.addr, &q_rope_tmp3_h1_5.shape, (q_rope_tmp3_h1_5.default_stride ? NULL : &q_rope_tmp3_h1_5.stride), (q_rope_tmp2_h1_5.default_stride ? NULL : &q_rope_tmp2_h1_5.stride), (sin_shared_10.default_stride ? NULL : &sin_shared_10.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_10.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_10.shape.n, .c = q_rope_h1_10.shape.c, .h = q_rope_h1_10.shape.h, .w = q_rope_h1_10.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_10.addr, q_rope_tmp0_h1_5.addr, q_rope_tmp3_h1_5.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_10.addr + 2, q_rope_tmp0_h1_5.addr + 2, q_rope_tmp1_h1_5.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_30 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_30.stride, 0, &q_nope_head_30.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_30 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_30.stride, 0, &q_wukv_acc_30.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_30 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_30.stride, 0, &q_wukv_head_30.shape, DT_BFP16);
    __ppl_tensor_info score_tile_20 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_20.stride, 0, &score_tile_20.shape, DT_FP32);
    __ppl_tensor_info prob_tile_20 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_20.stride, 0, &prob_tile_20.shape, DT_BFP16);
    __ppl_tensor_info scores_max_10 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_10.stride, 0, &scores_max_10.shape, DT_FP32);
    __ppl_tensor_info scores_sum_10 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_10.stride, 0, &scores_sum_10.shape, DT_FP32);
    __ppl_tensor_info work_score0_10 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_10.stride, 0, &work_score0_10.shape, DT_FP32);
    __ppl_tensor_info work_score1_10 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_10.stride, 0, &work_score1_10.shape, DT_FP32);
    __ppl_tensor_info coeff_10 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_10.stride, 0, &coeff_10.shape, DT_FP32);
    __ppl_tensor_info table_10 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_10.stride, 0, &table_10.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_20 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_20.stride, 0, &ctx_acc_20.shape, DT_FP32);
    __ppl_tensor_info ctx_head_30 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_30.stride, 0, &ctx_head_30.shape, DT_BFP16);
    __ppl_tensor_info out_part_30 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_30.stride, 0, &out_part_30.shape, DT_FP32);
    __ppl_tensor_info out_cast_30 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_30.stride, 0, &out_cast_30.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_23 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_20.stride, .addr = q_upper_h0_20.addr + ((0) * q_upper_h0_20.stride.c+(0) * q_upper_h0_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_20.stride.c+(0) * q_upper_h0_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_20.default_stride};
    __ppl_tensor_info q_nope_head_31 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_30.stride, .addr = q_nope_head_30.addr + ((0) * q_nope_head_30.stride.c+(0) * q_nope_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_30.stride.c+(0) * q_nope_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_30.default_stride};
    tpu_bdc_cpy(q_nope_head_31.addr, q_upper_h0_23.addr, &q_nope_head_31.shape, (q_nope_head_31.default_stride ? NULL : &q_nope_head_31.stride), (q_upper_h0_23.default_stride ? NULL : &q_upper_h0_23.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_30_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_30.addr, q_wukv_acc_30_scalar_f32, &q_wukv_acc_30.shape, (q_wukv_acc_30.default_stride ? NULL : &q_wukv_acc_30.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_32 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_30.stride, .addr = q_nope_head_30.addr + (((0) / 64) * q_nope_head_30.stride.c+(0) * q_nope_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_30.stride.c+(0) * q_nope_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_32 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_30.stride, .addr = wukv_full_tile_30.addr + (((0) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_31 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_30.stride, .addr = q_wukv_acc_30.addr + (((0) / 64) * q_wukv_acc_30.stride.c+(0) * q_wukv_acc_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_30.stride.c+(0) * q_wukv_acc_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_31.addr, q_nope_head_32.addr, wukv_full_tile_32.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_32 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_30.stride, .addr = q_wukv_acc_30.addr + ((0) * q_wukv_acc_30.stride.c+(0) * q_wukv_acc_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_30.stride.c+(0) * q_wukv_acc_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_30.default_stride};
    __ppl_tensor_info q_wukv_head_31 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_30.stride, .addr = q_wukv_head_30.addr + ((0) * q_wukv_head_30.stride.c+(0) * q_wukv_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_30.stride.c+(0) * q_wukv_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_30.default_stride};
    tpu_bdc_cast(q_wukv_head_31.addr, q_wukv_acc_32.addr, &q_wukv_head_31.shape, (q_wukv_head_31.default_stride ? NULL : &q_wukv_head_31.stride), (q_wukv_acc_32.default_stride ? NULL : &q_wukv_acc_32.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_32 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_30.stride, .addr = q_wukv_head_30.addr + (((0) / 64) * q_wukv_head_30.stride.c+(0) * q_wukv_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_30.stride.c+(0) * q_wukv_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_22 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_20.stride, .addr = kv_cache_full_20.addr + (((0) / 64) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_16 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_15.stride, .addr = score_latent_h0_15.addr + (((0) / 64) * score_latent_h0_15.stride.c+(0) * score_latent_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_15.stride.c+(0) * score_latent_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_16.addr, q_wukv_head_32.addr, kv_cache_full_22.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_11 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_10.stride, .addr = q_rope_h0_10.addr + (((0) / 64) * q_rope_h0_10.stride.c+(0) * q_rope_h0_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_10.stride.c+(0) * q_rope_h0_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_22 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_20.stride, .addr = pe_score_tile_20.addr + (((0) / 64) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_16 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_15.stride, .addr = score_rope_h0_15.addr + (((0) / 64) * score_rope_h0_15.stride.c+(0) * score_rope_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_15.stride.c+(0) * score_rope_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_16.addr, q_rope_h0_11.addr, pe_score_tile_22.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_20.addr, score_latent_h0_15.addr, score_rope_h0_15.addr, &score_tile_20.shape, (score_tile_20.default_stride ? NULL : &score_tile_20.stride), (score_latent_h0_15.default_stride ? NULL : &score_latent_h0_15.stride), (score_rope_h0_15.default_stride ? NULL : &score_rope_h0_15.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_20.addr, score_tile_20.addr, mask_row_10.addr, &score_tile_20.shape, (score_tile_20.default_stride ? NULL : &score_tile_20.stride), (score_tile_20.default_stride ? NULL : &score_tile_20.stride), (mask_row_10.default_stride ? NULL : &mask_row_10.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_20.addr, score_tile_20.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_20.shape, (score_tile_20.default_stride ? NULL : &score_tile_20.stride), (score_tile_20.default_stride ? NULL : &score_tile_20.stride), DT_FP32);
    {
    scalar_t scores_max_10_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_10.addr, scores_max_10_scalar_f32, &scores_max_10.shape, (scores_max_10.default_stride ? NULL : &scores_max_10.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_10 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_10.stride, 0, &tmp_buffer_max_10.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_20.shape.n, score_tile_20.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_20.shape.n, score_tile_20.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_20.shape.n, score_tile_20.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_20.shape.n, score_tile_20.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_20.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, score_tile_20.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_20.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_20.shape.w) {
          dim4 fill_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, align_w - score_tile_20.shape.w};
          int elem_size = 4;
          int offset = score_tile_20.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_20.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_20.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_20.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_10_stride;
    tpu_aligned_stride(&scores_max_10_stride, 0, &scores_max_10.shape, DT_FP32);
    scores_max_10_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_20.addr, score_tile_20.addr, scores_max_10.addr, &score_tile_20.shape, (score_tile_20.default_stride ? NULL : &score_tile_20.stride), (score_tile_20.default_stride ? NULL : &score_tile_20.stride), &scores_max_10_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_10.addr);
    tpu_bdc_load_fp32_exp_table(table_10.addr);
    tpu_bdc_fp32_exp(score_tile_20.addr, score_tile_20.addr, work_score0_10.addr, work_score1_10.addr, coeff_10.addr, table_10.addr, &score_tile_20.shape);
    __ppl_tensor_info tmp_buffer_sum_10 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_10.stride, 0, &tmp_buffer_sum_10.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_20.shape.n, score_tile_20.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_20.shape.n, score_tile_20.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_20.shape.n, score_tile_20.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_20.shape.n, score_tile_20.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_20.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, score_tile_20.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_20.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_20.shape.w) {
          dim4 fill_shape = {score_tile_20.shape.n, score_tile_20.shape.c, 1, align_w - score_tile_20.shape.w};
          int elem_size = 4;
          int offset = score_tile_20.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_20.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_20.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_20.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_10.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_21 = {.shape = {1, 1, 1, 192} , .stride = score_tile_20.stride, .addr = score_tile_20.addr + ((0) * score_tile_20.stride.c+(0) * score_tile_20.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_20.stride.c+(0) * score_tile_20.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_20.default_stride};
    __ppl_tensor_info prob_tile_21 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_20.stride, .addr = prob_tile_20.addr + ((0) * prob_tile_20.stride.c+(0) * prob_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_20.stride.c+(0) * prob_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_20.default_stride};
    tpu_bdc_cast(prob_tile_21.addr, score_tile_21.addr, &prob_tile_21.shape, (prob_tile_21.default_stride ? NULL : &prob_tile_21.stride), (score_tile_21.default_stride ? NULL : &score_tile_21.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_20_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_20.addr, ctx_acc_20_scalar_f32, &ctx_acc_20.shape, (ctx_acc_20.default_stride ? NULL : &ctx_acc_20.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_20.addr, prob_tile_20.addr, kv_cache_full_20.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_10_stride;
    tpu_aligned_stride(&scores_sum_10_stride, 0, &scores_sum_10.shape, DT_FP32);
    scores_sum_10_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_20.addr, ctx_acc_20.addr, scores_sum_10.addr, &ctx_acc_20.shape, (ctx_acc_20.default_stride ? NULL : &ctx_acc_20.stride), (ctx_acc_20.default_stride ? NULL : &ctx_acc_20.stride), &scores_sum_10_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_21 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_20.stride, .addr = ctx_acc_20.addr + ((0) * ctx_acc_20.stride.c+(0) * ctx_acc_20.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_20.stride.c+(0) * ctx_acc_20.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_20.default_stride};
    __ppl_tensor_info ctx_head_31 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_30.stride, .addr = ctx_head_30.addr + ((0) * ctx_head_30.stride.c+(0) * ctx_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_30.stride.c+(0) * ctx_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_30.default_stride};
    tpu_bdc_cast(ctx_head_31.addr, ctx_acc_21.addr, &ctx_head_31.shape, (ctx_head_31.default_stride ? NULL : &ctx_head_31.stride), (ctx_acc_21.default_stride ? NULL : &ctx_acc_21.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_30_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_30.addr, out_part_30_scalar_f32, &out_part_30.shape, (out_part_30.default_stride ? NULL : &out_part_30.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_32 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_30.stride, .addr = ctx_head_30.addr + (((0) / 64) * ctx_head_30.stride.c+(0) * ctx_head_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_30.stride.c+(0) * ctx_head_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_33 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_30.stride, .addr = wukv_full_tile_30.addr + (((128) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_31 = {.shape = {1, 1, 1, 128}, .stride = out_part_30.stride, .addr = out_part_30.addr + (((0) / 64) * out_part_30.stride.c+(0) * out_part_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_30.stride.c+(0) * out_part_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_31.addr, ctx_head_32.addr, wukv_full_tile_33.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_32 = {.shape = {1, 1, 1, 128} , .stride = out_part_30.stride, .addr = out_part_30.addr + ((0) * out_part_30.stride.c+(0) * out_part_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_30.stride.c+(0) * out_part_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_30.default_stride};
    __ppl_tensor_info out_cast_31 = {.shape = {1, 1, 1, 128} , .stride = out_cast_30.stride, .addr = out_cast_30.addr + ((0) * out_cast_30.stride.c+(0) * out_cast_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_30.stride.c+(0) * out_cast_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_30.default_stride};
    tpu_bdc_cast(out_cast_31.addr, out_part_32.addr, &out_cast_31.shape, (out_cast_31.default_stride ? NULL : &out_cast_31.stride), (out_part_32.default_stride ? NULL : &out_part_32.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_32 = {.shape = {1, 1, 1, 128} , .stride = out_cast_30.stride, .addr = out_cast_30.addr + ((0) * out_cast_30.stride.c+(0) * out_cast_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_30.stride.c+(0) * out_cast_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_30.default_stride};
    __ppl_tensor_info OUT_10 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1280) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1280) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_10.addr, out_cast_32.addr, &OUT_10.shape, (OUT_10.default_stride ? NULL : &OUT_10.stride), (out_cast_32.default_stride ? NULL : &out_cast_32.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_33 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_33.stride, 0, &q_nope_head_33.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_33 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_33.stride, 0, &q_wukv_acc_33.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_33 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_33.stride, 0, &q_wukv_head_33.shape, DT_BFP16);
    __ppl_tensor_info score_tile_22 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_22.stride, 0, &score_tile_22.shape, DT_FP32);
    __ppl_tensor_info prob_tile_22 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_22.stride, 0, &prob_tile_22.shape, DT_BFP16);
    __ppl_tensor_info scores_max_11 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_11.stride, 0, &scores_max_11.shape, DT_FP32);
    __ppl_tensor_info scores_sum_11 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_11.stride, 0, &scores_sum_11.shape, DT_FP32);
    __ppl_tensor_info work_score0_11 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_11.stride, 0, &work_score0_11.shape, DT_FP32);
    __ppl_tensor_info work_score1_11 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_11.stride, 0, &work_score1_11.shape, DT_FP32);
    __ppl_tensor_info coeff_11 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_11.stride, 0, &coeff_11.shape, DT_FP32);
    __ppl_tensor_info table_11 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_11.stride, 0, &table_11.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_22 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_22.stride, 0, &ctx_acc_22.shape, DT_FP32);
    __ppl_tensor_info ctx_head_33 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_33.stride, 0, &ctx_head_33.shape, DT_BFP16);
    __ppl_tensor_info out_part_33 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_33.stride, 0, &out_part_33.shape, DT_FP32);
    __ppl_tensor_info out_cast_33 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_33.stride, 0, &out_cast_33.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_23 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_20.stride, .addr = q_upper_h1_20.addr + ((0) * q_upper_h1_20.stride.c+(0) * q_upper_h1_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_20.stride.c+(0) * q_upper_h1_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_20.default_stride};
    __ppl_tensor_info q_nope_head_34 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_33.stride, .addr = q_nope_head_33.addr + ((0) * q_nope_head_33.stride.c+(0) * q_nope_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_33.stride.c+(0) * q_nope_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_33.default_stride};
    tpu_bdc_cpy(q_nope_head_34.addr, q_upper_h1_23.addr, &q_nope_head_34.shape, (q_nope_head_34.default_stride ? NULL : &q_nope_head_34.stride), (q_upper_h1_23.default_stride ? NULL : &q_upper_h1_23.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_33_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_33.addr, q_wukv_acc_33_scalar_f32, &q_wukv_acc_33.shape, (q_wukv_acc_33.default_stride ? NULL : &q_wukv_acc_33.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_35 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_33.stride, .addr = q_nope_head_33.addr + (((0) / 64) * q_nope_head_33.stride.c+(0) * q_nope_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_33.stride.c+(0) * q_nope_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_34 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_30.stride, .addr = wukv_full_tile_30.addr + (((256) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_34 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_33.stride, .addr = q_wukv_acc_33.addr + (((0) / 64) * q_wukv_acc_33.stride.c+(0) * q_wukv_acc_33.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_33.stride.c+(0) * q_wukv_acc_33.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_34.addr, q_nope_head_35.addr, wukv_full_tile_34.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_35 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_33.stride, .addr = q_wukv_acc_33.addr + ((0) * q_wukv_acc_33.stride.c+(0) * q_wukv_acc_33.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_33.stride.c+(0) * q_wukv_acc_33.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_33.default_stride};
    __ppl_tensor_info q_wukv_head_34 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_33.stride, .addr = q_wukv_head_33.addr + ((0) * q_wukv_head_33.stride.c+(0) * q_wukv_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_33.stride.c+(0) * q_wukv_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_33.default_stride};
    tpu_bdc_cast(q_wukv_head_34.addr, q_wukv_acc_35.addr, &q_wukv_head_34.shape, (q_wukv_head_34.default_stride ? NULL : &q_wukv_head_34.stride), (q_wukv_acc_35.default_stride ? NULL : &q_wukv_acc_35.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_35 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_33.stride, .addr = q_wukv_head_33.addr + (((0) / 64) * q_wukv_head_33.stride.c+(0) * q_wukv_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_33.stride.c+(0) * q_wukv_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_23 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_20.stride, .addr = kv_cache_full_20.addr + (((0) / 64) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_20.stride.c+(0) * kv_cache_full_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_17 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_15.stride, .addr = score_latent_h0_15.addr + (((0) / 64) * score_latent_h0_15.stride.c+(0) * score_latent_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_15.stride.c+(0) * score_latent_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_17.addr, q_wukv_head_35.addr, kv_cache_full_23.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_11 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_10.stride, .addr = q_rope_h1_10.addr + (((0) / 64) * q_rope_h1_10.stride.c+(0) * q_rope_h1_10.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_10.stride.c+(0) * q_rope_h1_10.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_23 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_20.stride, .addr = pe_score_tile_20.addr + (((0) / 64) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_20.stride.c+(0) * pe_score_tile_20.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_17 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_15.stride, .addr = score_rope_h0_15.addr + (((0) / 64) * score_rope_h0_15.stride.c+(0) * score_rope_h0_15.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_15.stride.c+(0) * score_rope_h0_15.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_17.addr, q_rope_h1_11.addr, pe_score_tile_23.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_22.addr, score_latent_h0_15.addr, score_rope_h0_15.addr, &score_tile_22.shape, (score_tile_22.default_stride ? NULL : &score_tile_22.stride), (score_latent_h0_15.default_stride ? NULL : &score_latent_h0_15.stride), (score_rope_h0_15.default_stride ? NULL : &score_rope_h0_15.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_22.addr, score_tile_22.addr, mask_row_10.addr, &score_tile_22.shape, (score_tile_22.default_stride ? NULL : &score_tile_22.stride), (score_tile_22.default_stride ? NULL : &score_tile_22.stride), (mask_row_10.default_stride ? NULL : &mask_row_10.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_22.addr, score_tile_22.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_22.shape, (score_tile_22.default_stride ? NULL : &score_tile_22.stride), (score_tile_22.default_stride ? NULL : &score_tile_22.stride), DT_FP32);
    {
    scalar_t scores_max_11_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_11.addr, scores_max_11_scalar_f32, &scores_max_11.shape, (scores_max_11.default_stride ? NULL : &scores_max_11.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_11 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_11.stride, 0, &tmp_buffer_max_11.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_22.shape.n, score_tile_22.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_22.shape.n, score_tile_22.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_22.shape.n, score_tile_22.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_22.shape.n, score_tile_22.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_22.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, score_tile_22.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_22.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_22.shape.w) {
          dim4 fill_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, align_w - score_tile_22.shape.w};
          int elem_size = 4;
          int offset = score_tile_22.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_22.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_22.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_22.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_11_stride;
    tpu_aligned_stride(&scores_max_11_stride, 0, &scores_max_11.shape, DT_FP32);
    scores_max_11_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_22.addr, score_tile_22.addr, scores_max_11.addr, &score_tile_22.shape, (score_tile_22.default_stride ? NULL : &score_tile_22.stride), (score_tile_22.default_stride ? NULL : &score_tile_22.stride), &scores_max_11_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_11.addr);
    tpu_bdc_load_fp32_exp_table(table_11.addr);
    tpu_bdc_fp32_exp(score_tile_22.addr, score_tile_22.addr, work_score0_11.addr, work_score1_11.addr, coeff_11.addr, table_11.addr, &score_tile_22.shape);
    __ppl_tensor_info tmp_buffer_sum_11 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_11.stride, 0, &tmp_buffer_sum_11.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_22.shape.n, score_tile_22.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_22.shape.n, score_tile_22.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_22.shape.n, score_tile_22.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_22.shape.n, score_tile_22.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_22.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, score_tile_22.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_22.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_22.shape.w) {
          dim4 fill_shape = {score_tile_22.shape.n, score_tile_22.shape.c, 1, align_w - score_tile_22.shape.w};
          int elem_size = 4;
          int offset = score_tile_22.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_22.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_22.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_22.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_11.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_23 = {.shape = {1, 1, 1, 192} , .stride = score_tile_22.stride, .addr = score_tile_22.addr + ((0) * score_tile_22.stride.c+(0) * score_tile_22.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_22.stride.c+(0) * score_tile_22.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_22.default_stride};
    __ppl_tensor_info prob_tile_23 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_22.stride, .addr = prob_tile_22.addr + ((0) * prob_tile_22.stride.c+(0) * prob_tile_22.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_22.stride.c+(0) * prob_tile_22.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_22.default_stride};
    tpu_bdc_cast(prob_tile_23.addr, score_tile_23.addr, &prob_tile_23.shape, (prob_tile_23.default_stride ? NULL : &prob_tile_23.stride), (score_tile_23.default_stride ? NULL : &score_tile_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_22_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_22.addr, ctx_acc_22_scalar_f32, &ctx_acc_22.shape, (ctx_acc_22.default_stride ? NULL : &ctx_acc_22.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_22.addr, prob_tile_22.addr, kv_cache_full_20.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_11_stride;
    tpu_aligned_stride(&scores_sum_11_stride, 0, &scores_sum_11.shape, DT_FP32);
    scores_sum_11_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_22.addr, ctx_acc_22.addr, scores_sum_11.addr, &ctx_acc_22.shape, (ctx_acc_22.default_stride ? NULL : &ctx_acc_22.stride), (ctx_acc_22.default_stride ? NULL : &ctx_acc_22.stride), &scores_sum_11_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_23 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_22.stride, .addr = ctx_acc_22.addr + ((0) * ctx_acc_22.stride.c+(0) * ctx_acc_22.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_22.stride.c+(0) * ctx_acc_22.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_22.default_stride};
    __ppl_tensor_info ctx_head_34 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_33.stride, .addr = ctx_head_33.addr + ((0) * ctx_head_33.stride.c+(0) * ctx_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_33.stride.c+(0) * ctx_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_33.default_stride};
    tpu_bdc_cast(ctx_head_34.addr, ctx_acc_23.addr, &ctx_head_34.shape, (ctx_head_34.default_stride ? NULL : &ctx_head_34.stride), (ctx_acc_23.default_stride ? NULL : &ctx_acc_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_33_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_33.addr, out_part_33_scalar_f32, &out_part_33.shape, (out_part_33.default_stride ? NULL : &out_part_33.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_35 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_33.stride, .addr = ctx_head_33.addr + (((0) / 64) * ctx_head_33.stride.c+(0) * ctx_head_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_33.stride.c+(0) * ctx_head_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_35 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_30.stride, .addr = wukv_full_tile_30.addr + (((384) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_30.stride.c+(0) * wukv_full_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_34 = {.shape = {1, 1, 1, 128}, .stride = out_part_33.stride, .addr = out_part_33.addr + (((0) / 64) * out_part_33.stride.c+(0) * out_part_33.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_33.stride.c+(0) * out_part_33.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_34.addr, ctx_head_35.addr, wukv_full_tile_35.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_35 = {.shape = {1, 1, 1, 128} , .stride = out_part_33.stride, .addr = out_part_33.addr + ((0) * out_part_33.stride.c+(0) * out_part_33.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_33.stride.c+(0) * out_part_33.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_33.default_stride};
    __ppl_tensor_info out_cast_34 = {.shape = {1, 1, 1, 128} , .stride = out_cast_33.stride, .addr = out_cast_33.addr + ((0) * out_cast_33.stride.c+(0) * out_cast_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_33.stride.c+(0) * out_cast_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_33.default_stride};
    tpu_bdc_cast(out_cast_34.addr, out_part_35.addr, &out_cast_34.shape, (out_cast_34.default_stride ? NULL : &out_cast_34.stride), (out_part_35.default_stride ? NULL : &out_part_35.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_35 = {.shape = {1, 1, 1, 128} , .stride = out_cast_33.stride, .addr = out_cast_33.addr + ((0) * out_cast_33.stride.c+(0) * out_cast_33.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_33.stride.c+(0) * out_cast_33.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_33.default_stride};
    __ppl_tensor_info OUT_11 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1408) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1408) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_11.addr, out_cast_35.addr, &OUT_11.shape, (OUT_11.default_stride ? NULL : &OUT_11.stride), (out_cast_35.default_stride ? NULL : &out_cast_35.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_18.stride, 0, &q_upper_acc_h0_18.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_24 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_24.stride, 0, &q_upper_h0_24.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_18.stride, 0, &q_upper_acc_h1_18.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_24 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_24.stride, 0, &q_upper_h1_24.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_12.stride, 0, &q_rope_src_h0_12.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_12.stride, 0, &q_rope_h0_12.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_6.stride, 0, &q_rope_tmp0_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_6.stride, 0, &q_rope_tmp1_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_6.stride, 0, &q_rope_tmp2_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_6.stride, 0, &q_rope_tmp3_h0_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_12.stride, 0, &q_rope_src_h1_12.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_12.stride, 0, &q_rope_h1_12.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_6.stride, 0, &q_rope_tmp0_h1_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_6.stride, 0, &q_rope_tmp1_h1_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_6.stride, 0, &q_rope_tmp2_h1_6.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_6.stride, 0, &q_rope_tmp3_h1_6.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_12.stride, 0, &cos_shared_12.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_12.stride, 0, &sin_shared_12.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_12.stride, 0, &pe_cur_12.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_6.stride, 0, &pe_cur_cos_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_6.stride, 0, &pe_cur_sin_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_6.stride, 0, &pe_cur_neg_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_6 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_6.stride, 0, &pe_cur_neg_sin_6.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_12 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_12.stride, 0, &pe_cur_rope_12.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_36 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_36.stride, 0, &wukv_full_tile_36.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_24 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_24.stride, 0, &kv_cache_full_24.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_24 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_24.stride, 0, &pe_score_tile_24.shape, DT_BFP16);
  __ppl_tensor_info mask_row_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_12.stride, 0, &mask_row_12.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_18.stride, 0, &score_latent_h0_18.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_18 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_18.stride, 0, &score_rope_h0_18.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_6.stride, 0, &score_latent_h1_6.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_6 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_6.stride, 0, &score_rope_h1_6.shape, DT_FP32);
  if (tpu_workitem_index() == 6) {
    __ppl_tensor_info COS_6 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_13 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_12.stride, .addr = cos_shared_12.addr + ((0) * cos_shared_12.stride.c+(0) * cos_shared_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_12.stride.c+(0) * cos_shared_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_12.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_13.addr, COS_6.addr, &cos_shared_13.shape, (cos_shared_13.default_stride ? NULL : &cos_shared_13.stride), (COS_6.default_stride ? NULL : &COS_6.stride), DT_BFP16);
    __ppl_tensor_info SIN_6 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_13 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_12.stride, .addr = sin_shared_12.addr + ((0) * sin_shared_12.stride.c+(0) * sin_shared_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_12.stride.c+(0) * sin_shared_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_12.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_13.addr, SIN_6.addr, &sin_shared_13.shape, (sin_shared_13.default_stride ? NULL : &sin_shared_13.stride), (SIN_6.default_stride ? NULL : &SIN_6.stride), DT_BFP16);
    for (int d_12 = 0; d_12 < 4; ++d_12) {
      __ppl_tensor_info kv_cur_tile_18 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_18.stride, 0, &kv_cur_tile_18.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_6 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_12 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_12 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_19 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_18.stride, .addr = kv_cur_tile_18.addr + ((0) * kv_cur_tile_18.stride.c+(0) * kv_cur_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_18.stride.c+(0) * kv_cur_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_18.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_19.addr, KV_CUR_6.addr, &kv_cur_tile_19.shape, (kv_cur_tile_19.default_stride ? NULL : &kv_cur_tile_19.stride), (KV_CUR_6.default_stride ? NULL : &KV_CUR_6.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_20 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_18.stride, .addr = kv_cur_tile_18.addr + ((0) * kv_cur_tile_18.stride.c+(0) * kv_cur_tile_18.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_18.stride.c+(0) * kv_cur_tile_18.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_18.default_stride};
      __ppl_tensor_info KVcache_OUT_12 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_12 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_12 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_12.addr, kv_cur_tile_20.addr, &KVcache_OUT_12.shape, (KVcache_OUT_12.default_stride ? NULL : &KVcache_OUT_12.stride), (kv_cur_tile_20.default_stride ? NULL : &kv_cur_tile_20.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_6 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_13 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_12.stride, .addr = pe_cur_12.addr + ((0) * pe_cur_12.stride.c+(0) * pe_cur_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_12.stride.c+(0) * pe_cur_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_12.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_13.addr, PE_CUR_6.addr, &pe_cur_13.shape, (pe_cur_13.default_stride ? NULL : &pe_cur_13.stride), (PE_CUR_6.default_stride ? NULL : &PE_CUR_6.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_6.addr, pe_cur_12.addr, cos_shared_12.addr, &pe_cur_cos_6.shape, (pe_cur_cos_6.default_stride ? NULL : &pe_cur_cos_6.stride), (pe_cur_12.default_stride ? NULL : &pe_cur_12.stride), (cos_shared_12.default_stride ? NULL : &cos_shared_12.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_6.addr, pe_cur_12.addr, sin_shared_12.addr, &pe_cur_sin_6.shape, (pe_cur_sin_6.default_stride ? NULL : &pe_cur_sin_6.stride), (pe_cur_12.default_stride ? NULL : &pe_cur_12.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_6_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_6_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_6_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_6.addr, pe_cur_12.addr, pe_cur_neg_6_scalar_DT_BFP16, &pe_cur_neg_6.shape, (pe_cur_neg_6.default_stride ? NULL : &pe_cur_neg_6.stride), (pe_cur_12.default_stride ? NULL : &pe_cur_12.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_6.addr, pe_cur_neg_6.addr, sin_shared_12.addr, &pe_cur_neg_sin_6.shape, (pe_cur_neg_sin_6.default_stride ? NULL : &pe_cur_neg_sin_6.stride), (pe_cur_neg_6.default_stride ? NULL : &pe_cur_neg_6.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_12.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_12.shape.n, .c = pe_cur_rope_12.shape.c, .h = pe_cur_rope_12.shape.h, .w = pe_cur_rope_12.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_12.addr, pe_cur_cos_6.addr, pe_cur_neg_sin_6.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_12.addr + 2, pe_cur_cos_6.addr + 2, pe_cur_sin_6.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_13 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_12.stride, .addr = pe_cur_rope_12.addr + ((0) * pe_cur_rope_12.stride.c+(0) * pe_cur_rope_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_12.stride.c+(0) * pe_cur_rope_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_12.default_stride};
    __ppl_tensor_info PEcache_OUT_12 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_12.addr, pe_cur_rope_13.addr, &PEcache_OUT_12.shape, (PEcache_OUT_12.default_stride ? NULL : &PEcache_OUT_12.stride), (pe_cur_rope_13.default_stride ? NULL : &pe_cur_rope_13.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_18.addr, q_upper_acc_h0_18_scalar_f32, &q_upper_acc_h0_18.shape, (q_upper_acc_h0_18.default_stride ? NULL : &q_upper_acc_h0_18.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_18.addr, q_upper_acc_h1_18_scalar_f32, &q_upper_acc_h1_18.shape, (q_upper_acc_h1_18.default_stride ? NULL : &q_upper_acc_h1_18.stride), DT_FP32);
    }
    for (int k_6 = 0; k_6 < 12; ++k_6) {
      __ppl_tensor_info q_tile_24 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_24.stride, 0, &q_tile_24.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_18 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_18.stride, 0, &wuq_fp8_18.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_48 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_48.stride, 0, &wuq_scale_stage_48.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_24 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_24.stride, 0, &wuq_tile_24.shape, DT_BFP16);
      __ppl_tensor_info Q_6 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_6 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_6 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_25 = {.shape = {1, 1, 1, 128} , .stride = q_tile_24.stride, .addr = q_tile_24.addr + ((0) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_24.default_stride};
      tpu_gdma_cpy_S2L(q_tile_25.addr, Q_6.addr, &q_tile_25.shape, (q_tile_25.default_stride ? NULL : &q_tile_25.stride), (Q_6.default_stride ? NULL : &Q_6.stride), DT_BFP16);
      __ppl_tensor_info WUQ_6 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((2304) * 1536+((k_6 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((2304) * 1536+((k_6 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_19 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_18.stride, .addr = wuq_fp8_18.addr + ((0) * wuq_fp8_18.stride.c+(0) * wuq_fp8_18.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_18.stride.c+(0) * wuq_fp8_18.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_18.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_19.addr, WUQ_6.addr, &wuq_fp8_19.shape, (wuq_fp8_19.default_stride ? NULL : &wuq_fp8_19.stride), (WUQ_6.default_stride ? NULL : &WUQ_6.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_108 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_108.stride, 0, &wuq_scale_scalar_108.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_36 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((18) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((18) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_109 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_108.stride, .addr = wuq_scale_scalar_108.addr + ((0) * wuq_scale_scalar_108.stride.c+(0) * wuq_scale_scalar_108.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_108.stride.c+(0) * wuq_scale_scalar_108.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_108.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_109.addr, WUQ_SCALE_36.addr, &wuq_scale_scalar_109.shape, (wuq_scale_scalar_109.default_stride ? NULL : &wuq_scale_scalar_109.stride), (WUQ_SCALE_36.default_stride ? NULL : &WUQ_SCALE_36.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_110 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_108.stride, .addr = wuq_scale_scalar_108.addr + (((0) / 64) * wuq_scale_scalar_108.stride.c+(0) * wuq_scale_scalar_108.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_108.stride.c+(0) * wuq_scale_scalar_108.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_49 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((0) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_49.addr, wuq_scale_scalar_110.addr, &wuq_scale_stage_49.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_111 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_111.stride, 0, &wuq_scale_scalar_111.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_37 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((18) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((18) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_112 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_111.stride, .addr = wuq_scale_scalar_111.addr + ((0) * wuq_scale_scalar_111.stride.c+(0) * wuq_scale_scalar_111.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_111.stride.c+(0) * wuq_scale_scalar_111.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_111.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_112.addr, WUQ_SCALE_37.addr, &wuq_scale_scalar_112.shape, (wuq_scale_scalar_112.default_stride ? NULL : &wuq_scale_scalar_112.stride), (WUQ_SCALE_37.default_stride ? NULL : &WUQ_SCALE_37.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_113 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_111.stride, .addr = wuq_scale_scalar_111.addr + (((0) / 64) * wuq_scale_scalar_111.stride.c+(0) * wuq_scale_scalar_111.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_111.stride.c+(0) * wuq_scale_scalar_111.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_50 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((64) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_50.addr, wuq_scale_scalar_113.addr, &wuq_scale_stage_50.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_114 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_114.stride, 0, &wuq_scale_scalar_114.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_38 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((19) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((19) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_115 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_114.stride, .addr = wuq_scale_scalar_114.addr + ((0) * wuq_scale_scalar_114.stride.c+(0) * wuq_scale_scalar_114.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_114.stride.c+(0) * wuq_scale_scalar_114.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_114.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_115.addr, WUQ_SCALE_38.addr, &wuq_scale_scalar_115.shape, (wuq_scale_scalar_115.default_stride ? NULL : &wuq_scale_scalar_115.stride), (WUQ_SCALE_38.default_stride ? NULL : &WUQ_SCALE_38.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_116 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_114.stride, .addr = wuq_scale_scalar_114.addr + (((0) / 64) * wuq_scale_scalar_114.stride.c+(0) * wuq_scale_scalar_114.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_114.stride.c+(0) * wuq_scale_scalar_114.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_51 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((128) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_51.addr, wuq_scale_scalar_116.addr, &wuq_scale_stage_51.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_117 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_117.stride, 0, &wuq_scale_scalar_117.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_39 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((19) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((19) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_118 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_117.stride, .addr = wuq_scale_scalar_117.addr + ((0) * wuq_scale_scalar_117.stride.c+(0) * wuq_scale_scalar_117.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_117.stride.c+(0) * wuq_scale_scalar_117.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_117.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_118.addr, WUQ_SCALE_39.addr, &wuq_scale_scalar_118.shape, (wuq_scale_scalar_118.default_stride ? NULL : &wuq_scale_scalar_118.stride), (WUQ_SCALE_39.default_stride ? NULL : &WUQ_SCALE_39.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_119 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_117.stride, .addr = wuq_scale_scalar_117.addr + (((0) / 64) * wuq_scale_scalar_117.stride.c+(0) * wuq_scale_scalar_117.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_117.stride.c+(0) * wuq_scale_scalar_117.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_52 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((192) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_52.addr, wuq_scale_scalar_119.addr, &wuq_scale_stage_52.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_120 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_120.stride, 0, &wuq_scale_scalar_120.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_40 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((20) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((20) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_121 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_120.stride, .addr = wuq_scale_scalar_120.addr + ((0) * wuq_scale_scalar_120.stride.c+(0) * wuq_scale_scalar_120.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_120.stride.c+(0) * wuq_scale_scalar_120.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_120.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_121.addr, WUQ_SCALE_40.addr, &wuq_scale_scalar_121.shape, (wuq_scale_scalar_121.default_stride ? NULL : &wuq_scale_scalar_121.stride), (WUQ_SCALE_40.default_stride ? NULL : &WUQ_SCALE_40.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_122 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_120.stride, .addr = wuq_scale_scalar_120.addr + (((0) / 64) * wuq_scale_scalar_120.stride.c+(0) * wuq_scale_scalar_120.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_120.stride.c+(0) * wuq_scale_scalar_120.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_53 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((256) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_53.addr, wuq_scale_scalar_122.addr, &wuq_scale_stage_53.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_123 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_123.stride, 0, &wuq_scale_scalar_123.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_41 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((20) * 12+(k_6) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((20) * 12+(k_6) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_124 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_123.stride, .addr = wuq_scale_scalar_123.addr + ((0) * wuq_scale_scalar_123.stride.c+(0) * wuq_scale_scalar_123.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_123.stride.c+(0) * wuq_scale_scalar_123.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_123.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_124.addr, WUQ_SCALE_41.addr, &wuq_scale_scalar_124.shape, (wuq_scale_scalar_124.default_stride ? NULL : &wuq_scale_scalar_124.stride), (WUQ_SCALE_41.default_stride ? NULL : &WUQ_SCALE_41.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_125 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_123.stride, .addr = wuq_scale_scalar_123.addr + (((0) / 64) * wuq_scale_scalar_123.stride.c+(0) * wuq_scale_scalar_123.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_123.stride.c+(0) * wuq_scale_scalar_123.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_54 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((320) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_54.addr, wuq_scale_scalar_125.addr, &wuq_scale_stage_54.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_20 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_18.stride, .addr = wuq_fp8_18.addr + (((0) / 64) * wuq_fp8_18.stride.c+(0) * wuq_fp8_18.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_18.stride.c+(0) * wuq_fp8_18.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_55 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_48.stride, .addr = wuq_scale_stage_48.addr + (((0) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_48.stride.c+(0) * wuq_scale_stage_48.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_25 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_24.stride, .addr = wuq_tile_24.addr + (((0) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_25.addr, wuq_fp8_20.addr, &wuq_tile_25.shape, (wuq_tile_25.default_stride ? NULL : &wuq_tile_25.stride), (wuq_fp8_20.default_stride ? NULL : &wuq_fp8_20.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_55_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_55_block_stride, 0, &wuq_scale_stage_55.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_55_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_25.addr, wuq_tile_25.addr, wuq_scale_stage_55.addr, &wuq_tile_25.shape, (wuq_tile_25.default_stride ? NULL : &wuq_tile_25.stride), (wuq_tile_25.default_stride ? NULL : &wuq_tile_25.stride), &wuq_scale_stage_55_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_26 = {.shape = {1, 1, 1, 128}, .stride = q_tile_24.stride, .addr = q_tile_24.addr + (((0) / 64) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_26 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_24.stride, .addr = wuq_tile_24.addr + (((0) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_19 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_18.stride, .addr = q_upper_acc_h0_18.addr + (((0) / 64) * q_upper_acc_h0_18.stride.c+(0) * q_upper_acc_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_18.stride.c+(0) * q_upper_acc_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_19.addr, q_tile_26.addr, wuq_tile_26.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_27 = {.shape = {1, 1, 1, 128}, .stride = q_tile_24.stride, .addr = q_tile_24.addr + (((0) / 64) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_24.stride.c+(0) * q_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_27 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_24.stride, .addr = wuq_tile_24.addr + (((192) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_24.stride.c+(0) * wuq_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_19 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_18.stride, .addr = q_upper_acc_h1_18.addr + (((0) / 64) * q_upper_acc_h1_18.stride.c+(0) * q_upper_acc_h1_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_18.stride.c+(0) * q_upper_acc_h1_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_19.addr, q_tile_27.addr, wuq_tile_27.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_20 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_18.stride, .addr = q_upper_acc_h0_18.addr + ((0) * q_upper_acc_h0_18.stride.c+(0) * q_upper_acc_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_18.stride.c+(0) * q_upper_acc_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_18.default_stride};
    __ppl_tensor_info q_upper_h0_25 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_24.stride, .addr = q_upper_h0_24.addr + ((0) * q_upper_h0_24.stride.c+(0) * q_upper_h0_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_24.stride.c+(0) * q_upper_h0_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_24.default_stride};
    tpu_bdc_cast(q_upper_h0_25.addr, q_upper_acc_h0_20.addr, &q_upper_h0_25.shape, (q_upper_h0_25.default_stride ? NULL : &q_upper_h0_25.stride), (q_upper_acc_h0_20.default_stride ? NULL : &q_upper_acc_h0_20.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_20 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_18.stride, .addr = q_upper_acc_h1_18.addr + ((0) * q_upper_acc_h1_18.stride.c+(0) * q_upper_acc_h1_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_18.stride.c+(0) * q_upper_acc_h1_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_18.default_stride};
    __ppl_tensor_info q_upper_h1_25 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_24.stride, .addr = q_upper_h1_24.addr + ((0) * q_upper_h1_24.stride.c+(0) * q_upper_h1_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_24.stride.c+(0) * q_upper_h1_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_24.default_stride};
    tpu_bdc_cast(q_upper_h1_25.addr, q_upper_acc_h1_20.addr, &q_upper_h1_25.shape, (q_upper_h1_25.default_stride ? NULL : &q_upper_h1_25.stride), (q_upper_acc_h1_20.default_stride ? NULL : &q_upper_acc_h1_20.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_13 = 0; d_13 < 4; ++d_13) {
      __ppl_tensor_info wukv_fp8_18 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_18.stride, 0, &wukv_fp8_18.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_36 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_36.stride, 0, &wukv_scale_stage_36.shape, DT_BFP16);
      __ppl_tensor_info WUKV_6 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((3072) * 512+((d_13 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((3072) * 512+((d_13 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_19 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_18.stride, .addr = wukv_fp8_18.addr + ((0) * wukv_fp8_18.stride.c+(0) * wukv_fp8_18.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_18.stride.c+(0) * wukv_fp8_18.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_18.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_19.addr, WUKV_6.addr, &wukv_fp8_19.shape, (wukv_fp8_19.default_stride ? NULL : &wukv_fp8_19.stride), (WUKV_6.default_stride ? NULL : &WUKV_6.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_60 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_60.stride, 0, &wukv_scale_scalar_60.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_24 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((24) * 4+(d_13) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((24) * 4+(d_13) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_61 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_60.stride, .addr = wukv_scale_scalar_60.addr + ((0) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_60.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_61.addr, WUKV_SCALE_24.addr, &wukv_scale_scalar_61.shape, (wukv_scale_scalar_61.default_stride ? NULL : &wukv_scale_scalar_61.stride), (WUKV_SCALE_24.default_stride ? NULL : &WUKV_SCALE_24.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_62 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_60.stride, .addr = wukv_scale_scalar_60.addr + (((0) / 64) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_37 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_36.stride, .addr = wukv_scale_stage_36.addr + (((0) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_37_chunk = wukv_scale_stage_37;
      wukv_scale_stage_37_chunk.shape.c = 64;
      wukv_scale_stage_37_chunk.addr = wukv_scale_stage_37.addr + ((0) * wukv_scale_stage_37.stride.c) * 2;
      wukv_scale_stage_37_chunk.offset = wukv_scale_stage_37.offset + ((0) * wukv_scale_stage_37.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_37_chunk.addr, wukv_scale_scalar_62.addr, &wukv_scale_stage_37_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_37_chunk_1 = wukv_scale_stage_37;
      wukv_scale_stage_37_chunk_1.shape.c = 64;
      wukv_scale_stage_37_chunk_1.addr = wukv_scale_stage_37.addr + ((1) * wukv_scale_stage_37.stride.c) * 2;
      wukv_scale_stage_37_chunk_1.offset = wukv_scale_stage_37.offset + ((1) * wukv_scale_stage_37.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_37_chunk_1.addr, wukv_scale_scalar_62.addr, &wukv_scale_stage_37_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_25 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((25) * 4+(d_13) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((25) * 4+(d_13) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_63 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_60.stride, .addr = wukv_scale_scalar_60.addr + ((0) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_60.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_63.addr, WUKV_SCALE_25.addr, &wukv_scale_scalar_63.shape, (wukv_scale_scalar_63.default_stride ? NULL : &wukv_scale_scalar_63.stride), (WUKV_SCALE_25.default_stride ? NULL : &WUKV_SCALE_25.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_64 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_60.stride, .addr = wukv_scale_scalar_60.addr + (((0) / 64) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_60.stride.c+(0) * wukv_scale_scalar_60.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_38 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_36.stride, .addr = wukv_scale_stage_36.addr + (((128) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_38_chunk = wukv_scale_stage_38;
      wukv_scale_stage_38_chunk.shape.c = 64;
      wukv_scale_stage_38_chunk.addr = wukv_scale_stage_38.addr + ((0) * wukv_scale_stage_38.stride.c) * 2;
      wukv_scale_stage_38_chunk.offset = wukv_scale_stage_38.offset + ((0) * wukv_scale_stage_38.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_38_chunk.addr, wukv_scale_scalar_64.addr, &wukv_scale_stage_38_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_38_chunk_1 = wukv_scale_stage_38;
      wukv_scale_stage_38_chunk_1.shape.c = 64;
      wukv_scale_stage_38_chunk_1.addr = wukv_scale_stage_38.addr + ((1) * wukv_scale_stage_38.stride.c) * 2;
      wukv_scale_stage_38_chunk_1.offset = wukv_scale_stage_38.offset + ((1) * wukv_scale_stage_38.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_38_chunk_1.addr, wukv_scale_scalar_64.addr, &wukv_scale_stage_38_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_65 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_65.stride, 0, &wukv_scale_scalar_65.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_26 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((26) * 4+(d_13) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((26) * 4+(d_13) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_66 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_65.stride, .addr = wukv_scale_scalar_65.addr + ((0) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_65.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_66.addr, WUKV_SCALE_26.addr, &wukv_scale_scalar_66.shape, (wukv_scale_scalar_66.default_stride ? NULL : &wukv_scale_scalar_66.stride), (WUKV_SCALE_26.default_stride ? NULL : &WUKV_SCALE_26.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_67 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_65.stride, .addr = wukv_scale_scalar_65.addr + (((0) / 64) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_39 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_36.stride, .addr = wukv_scale_stage_36.addr + (((256) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_39_chunk = wukv_scale_stage_39;
      wukv_scale_stage_39_chunk.shape.c = 64;
      wukv_scale_stage_39_chunk.addr = wukv_scale_stage_39.addr + ((0) * wukv_scale_stage_39.stride.c) * 2;
      wukv_scale_stage_39_chunk.offset = wukv_scale_stage_39.offset + ((0) * wukv_scale_stage_39.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_39_chunk.addr, wukv_scale_scalar_67.addr, &wukv_scale_stage_39_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_39_chunk_1 = wukv_scale_stage_39;
      wukv_scale_stage_39_chunk_1.shape.c = 64;
      wukv_scale_stage_39_chunk_1.addr = wukv_scale_stage_39.addr + ((1) * wukv_scale_stage_39.stride.c) * 2;
      wukv_scale_stage_39_chunk_1.offset = wukv_scale_stage_39.offset + ((1) * wukv_scale_stage_39.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_39_chunk_1.addr, wukv_scale_scalar_67.addr, &wukv_scale_stage_39_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_27 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((27) * 4+(d_13) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((27) * 4+(d_13) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_68 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_65.stride, .addr = wukv_scale_scalar_65.addr + ((0) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_65.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_68.addr, WUKV_SCALE_27.addr, &wukv_scale_scalar_68.shape, (wukv_scale_scalar_68.default_stride ? NULL : &wukv_scale_scalar_68.stride), (WUKV_SCALE_27.default_stride ? NULL : &WUKV_SCALE_27.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_69 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_65.stride, .addr = wukv_scale_scalar_65.addr + (((0) / 64) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_65.stride.c+(0) * wukv_scale_scalar_65.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_40 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_36.stride, .addr = wukv_scale_stage_36.addr + (((384) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_40_chunk = wukv_scale_stage_40;
      wukv_scale_stage_40_chunk.shape.c = 64;
      wukv_scale_stage_40_chunk.addr = wukv_scale_stage_40.addr + ((0) * wukv_scale_stage_40.stride.c) * 2;
      wukv_scale_stage_40_chunk.offset = wukv_scale_stage_40.offset + ((0) * wukv_scale_stage_40.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_40_chunk.addr, wukv_scale_scalar_69.addr, &wukv_scale_stage_40_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_40_chunk_1 = wukv_scale_stage_40;
      wukv_scale_stage_40_chunk_1.shape.c = 64;
      wukv_scale_stage_40_chunk_1.addr = wukv_scale_stage_40.addr + ((1) * wukv_scale_stage_40.stride.c) * 2;
      wukv_scale_stage_40_chunk_1.offset = wukv_scale_stage_40.offset + ((1) * wukv_scale_stage_40.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_40_chunk_1.addr, wukv_scale_scalar_69.addr, &wukv_scale_stage_40_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_20 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_18.stride, .addr = wukv_fp8_18.addr + (((0) / 64) * wukv_fp8_18.stride.c+(0) * wukv_fp8_18.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_18.stride.c+(0) * wukv_fp8_18.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_41 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_36.stride, .addr = wukv_scale_stage_36.addr + (((0) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_36.stride.c+(0) * wukv_scale_stage_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_37 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_36.stride, .addr = wukv_full_tile_36.addr + (((0) / 64) * wukv_full_tile_36.stride.c+((d_13 * 128)) * wukv_full_tile_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_36.stride.c+((d_13 * 128)) * wukv_full_tile_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_37.addr, wukv_fp8_20.addr, &wukv_full_tile_37.shape, (wukv_full_tile_37.default_stride ? NULL : &wukv_full_tile_37.stride), (wukv_fp8_20.default_stride ? NULL : &wukv_fp8_20.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_41_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_41_block_stride, 0, &wukv_scale_stage_41.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_41_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_37.addr, wukv_full_tile_37.addr, wukv_scale_stage_41.addr, &wukv_full_tile_37.shape, (wukv_full_tile_37.default_stride ? NULL : &wukv_full_tile_37.stride), (wukv_full_tile_37.default_stride ? NULL : &wukv_full_tile_37.stride), &wukv_scale_stage_41_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_13 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_25 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_24.stride, .addr = kv_cache_full_24.addr + ((0) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_24.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_25.addr, KVcache_OUT_13.addr, &kv_cache_full_25.shape, (kv_cache_full_25.default_stride ? NULL : &kv_cache_full_25.stride), (KVcache_OUT_13.default_stride ? NULL : &KVcache_OUT_13.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_13 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_25 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_24.stride, .addr = pe_score_tile_24.addr + ((0) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_24.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_25.addr, PEcache_OUT_13.addr, &pe_score_tile_25.shape, (pe_score_tile_25.default_stride ? NULL : &pe_score_tile_25.stride), (PEcache_OUT_13.default_stride ? NULL : &PEcache_OUT_13.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_6 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_13 = {.shape = {1, 1, 1, 192} , .stride = mask_row_12.stride, .addr = mask_row_12.addr + ((0) * mask_row_12.stride.c+(0) * mask_row_12.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_12.stride.c+(0) * mask_row_12.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_12.default_stride};
    tpu_gdma_cpy_S2L(mask_row_13.addr, DECODE_MASK_6.addr, &mask_row_13.shape, (mask_row_13.default_stride ? NULL : &mask_row_13.stride), (DECODE_MASK_6.default_stride ? NULL : &DECODE_MASK_6.stride), DT_FP32);
    {
    scalar_t score_latent_h0_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_18.addr, score_latent_h0_18_scalar_f32, &score_latent_h0_18.shape, (score_latent_h0_18.default_stride ? NULL : &score_latent_h0_18.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_18_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_18.addr, score_rope_h0_18_scalar_f32, &score_rope_h0_18.shape, (score_rope_h0_18.default_stride ? NULL : &score_rope_h0_18.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_6.addr, score_latent_h1_6_scalar_f32, &score_latent_h1_6.shape, (score_latent_h1_6.default_stride ? NULL : &score_latent_h1_6.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_6_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_6.addr, score_rope_h1_6_scalar_f32, &score_rope_h1_6.shape, (score_rope_h1_6.default_stride ? NULL : &score_rope_h1_6.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_26 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_24.stride, .addr = q_upper_h0_24.addr + ((0) * q_upper_h0_24.stride.c+(128) * q_upper_h0_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_24.stride.c+(128) * q_upper_h0_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_24.default_stride};
    __ppl_tensor_info q_rope_src_h0_13 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_12.stride, .addr = q_rope_src_h0_12.addr + ((0) * q_rope_src_h0_12.stride.c+(0) * q_rope_src_h0_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_12.stride.c+(0) * q_rope_src_h0_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_12.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_13.addr, q_upper_h0_26.addr, &q_rope_src_h0_13.shape, (q_rope_src_h0_13.default_stride ? NULL : &q_rope_src_h0_13.stride), (q_upper_h0_26.default_stride ? NULL : &q_upper_h0_26.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_6.addr, q_rope_src_h0_12.addr, cos_shared_12.addr, &q_rope_tmp0_h0_6.shape, (q_rope_tmp0_h0_6.default_stride ? NULL : &q_rope_tmp0_h0_6.stride), (q_rope_src_h0_12.default_stride ? NULL : &q_rope_src_h0_12.stride), (cos_shared_12.default_stride ? NULL : &cos_shared_12.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_6.addr, q_rope_src_h0_12.addr, sin_shared_12.addr, &q_rope_tmp1_h0_6.shape, (q_rope_tmp1_h0_6.default_stride ? NULL : &q_rope_tmp1_h0_6.stride), (q_rope_src_h0_12.default_stride ? NULL : &q_rope_src_h0_12.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_6_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_6_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_6_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_6.addr, q_rope_src_h0_12.addr, q_rope_tmp2_h0_6_scalar_DT_BFP16, &q_rope_tmp2_h0_6.shape, (q_rope_tmp2_h0_6.default_stride ? NULL : &q_rope_tmp2_h0_6.stride), (q_rope_src_h0_12.default_stride ? NULL : &q_rope_src_h0_12.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_6.addr, q_rope_tmp2_h0_6.addr, sin_shared_12.addr, &q_rope_tmp3_h0_6.shape, (q_rope_tmp3_h0_6.default_stride ? NULL : &q_rope_tmp3_h0_6.stride), (q_rope_tmp2_h0_6.default_stride ? NULL : &q_rope_tmp2_h0_6.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_12.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_12.shape.n, .c = q_rope_h0_12.shape.c, .h = q_rope_h0_12.shape.h, .w = q_rope_h0_12.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_12.addr, q_rope_tmp0_h0_6.addr, q_rope_tmp3_h0_6.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_12.addr + 2, q_rope_tmp0_h0_6.addr + 2, q_rope_tmp1_h0_6.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_26 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_24.stride, .addr = q_upper_h1_24.addr + ((0) * q_upper_h1_24.stride.c+(128) * q_upper_h1_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_24.stride.c+(128) * q_upper_h1_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_24.default_stride};
    __ppl_tensor_info q_rope_src_h1_13 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_12.stride, .addr = q_rope_src_h1_12.addr + ((0) * q_rope_src_h1_12.stride.c+(0) * q_rope_src_h1_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_12.stride.c+(0) * q_rope_src_h1_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_12.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_13.addr, q_upper_h1_26.addr, &q_rope_src_h1_13.shape, (q_rope_src_h1_13.default_stride ? NULL : &q_rope_src_h1_13.stride), (q_upper_h1_26.default_stride ? NULL : &q_upper_h1_26.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_6.addr, q_rope_src_h1_12.addr, cos_shared_12.addr, &q_rope_tmp0_h1_6.shape, (q_rope_tmp0_h1_6.default_stride ? NULL : &q_rope_tmp0_h1_6.stride), (q_rope_src_h1_12.default_stride ? NULL : &q_rope_src_h1_12.stride), (cos_shared_12.default_stride ? NULL : &cos_shared_12.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_6.addr, q_rope_src_h1_12.addr, sin_shared_12.addr, &q_rope_tmp1_h1_6.shape, (q_rope_tmp1_h1_6.default_stride ? NULL : &q_rope_tmp1_h1_6.stride), (q_rope_src_h1_12.default_stride ? NULL : &q_rope_src_h1_12.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_6_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_6_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_6_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_6.addr, q_rope_src_h1_12.addr, q_rope_tmp2_h1_6_scalar_DT_BFP16, &q_rope_tmp2_h1_6.shape, (q_rope_tmp2_h1_6.default_stride ? NULL : &q_rope_tmp2_h1_6.stride), (q_rope_src_h1_12.default_stride ? NULL : &q_rope_src_h1_12.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_6.addr, q_rope_tmp2_h1_6.addr, sin_shared_12.addr, &q_rope_tmp3_h1_6.shape, (q_rope_tmp3_h1_6.default_stride ? NULL : &q_rope_tmp3_h1_6.stride), (q_rope_tmp2_h1_6.default_stride ? NULL : &q_rope_tmp2_h1_6.stride), (sin_shared_12.default_stride ? NULL : &sin_shared_12.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_12.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_12.shape.n, .c = q_rope_h1_12.shape.c, .h = q_rope_h1_12.shape.h, .w = q_rope_h1_12.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_12.addr, q_rope_tmp0_h1_6.addr, q_rope_tmp3_h1_6.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_12.addr + 2, q_rope_tmp0_h1_6.addr + 2, q_rope_tmp1_h1_6.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_36 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_36.stride, 0, &q_nope_head_36.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_36 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_36.stride, 0, &q_wukv_acc_36.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_36 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_36.stride, 0, &q_wukv_head_36.shape, DT_BFP16);
    __ppl_tensor_info score_tile_24 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_24.stride, 0, &score_tile_24.shape, DT_FP32);
    __ppl_tensor_info prob_tile_24 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_24.stride, 0, &prob_tile_24.shape, DT_BFP16);
    __ppl_tensor_info scores_max_12 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_12.stride, 0, &scores_max_12.shape, DT_FP32);
    __ppl_tensor_info scores_sum_12 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_12.stride, 0, &scores_sum_12.shape, DT_FP32);
    __ppl_tensor_info work_score0_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_12.stride, 0, &work_score0_12.shape, DT_FP32);
    __ppl_tensor_info work_score1_12 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_12.stride, 0, &work_score1_12.shape, DT_FP32);
    __ppl_tensor_info coeff_12 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_12.stride, 0, &coeff_12.shape, DT_FP32);
    __ppl_tensor_info table_12 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_12.stride, 0, &table_12.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_24 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_24.stride, 0, &ctx_acc_24.shape, DT_FP32);
    __ppl_tensor_info ctx_head_36 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_36.stride, 0, &ctx_head_36.shape, DT_BFP16);
    __ppl_tensor_info out_part_36 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_36.stride, 0, &out_part_36.shape, DT_FP32);
    __ppl_tensor_info out_cast_36 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_36.stride, 0, &out_cast_36.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_27 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_24.stride, .addr = q_upper_h0_24.addr + ((0) * q_upper_h0_24.stride.c+(0) * q_upper_h0_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_24.stride.c+(0) * q_upper_h0_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_24.default_stride};
    __ppl_tensor_info q_nope_head_37 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_36.stride, .addr = q_nope_head_36.addr + ((0) * q_nope_head_36.stride.c+(0) * q_nope_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_36.stride.c+(0) * q_nope_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_36.default_stride};
    tpu_bdc_cpy(q_nope_head_37.addr, q_upper_h0_27.addr, &q_nope_head_37.shape, (q_nope_head_37.default_stride ? NULL : &q_nope_head_37.stride), (q_upper_h0_27.default_stride ? NULL : &q_upper_h0_27.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_36_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_36.addr, q_wukv_acc_36_scalar_f32, &q_wukv_acc_36.shape, (q_wukv_acc_36.default_stride ? NULL : &q_wukv_acc_36.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_38 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_36.stride, .addr = q_nope_head_36.addr + (((0) / 64) * q_nope_head_36.stride.c+(0) * q_nope_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_36.stride.c+(0) * q_nope_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_38 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_36.stride, .addr = wukv_full_tile_36.addr + (((0) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_37 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_36.stride, .addr = q_wukv_acc_36.addr + (((0) / 64) * q_wukv_acc_36.stride.c+(0) * q_wukv_acc_36.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_36.stride.c+(0) * q_wukv_acc_36.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_37.addr, q_nope_head_38.addr, wukv_full_tile_38.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_38 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_36.stride, .addr = q_wukv_acc_36.addr + ((0) * q_wukv_acc_36.stride.c+(0) * q_wukv_acc_36.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_36.stride.c+(0) * q_wukv_acc_36.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_36.default_stride};
    __ppl_tensor_info q_wukv_head_37 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_36.stride, .addr = q_wukv_head_36.addr + ((0) * q_wukv_head_36.stride.c+(0) * q_wukv_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_36.stride.c+(0) * q_wukv_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_36.default_stride};
    tpu_bdc_cast(q_wukv_head_37.addr, q_wukv_acc_38.addr, &q_wukv_head_37.shape, (q_wukv_head_37.default_stride ? NULL : &q_wukv_head_37.stride), (q_wukv_acc_38.default_stride ? NULL : &q_wukv_acc_38.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_38 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_36.stride, .addr = q_wukv_head_36.addr + (((0) / 64) * q_wukv_head_36.stride.c+(0) * q_wukv_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_36.stride.c+(0) * q_wukv_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_26 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_24.stride, .addr = kv_cache_full_24.addr + (((0) / 64) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_19 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_18.stride, .addr = score_latent_h0_18.addr + (((0) / 64) * score_latent_h0_18.stride.c+(0) * score_latent_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_18.stride.c+(0) * score_latent_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_19.addr, q_wukv_head_38.addr, kv_cache_full_26.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_13 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_12.stride, .addr = q_rope_h0_12.addr + (((0) / 64) * q_rope_h0_12.stride.c+(0) * q_rope_h0_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_12.stride.c+(0) * q_rope_h0_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_26 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_24.stride, .addr = pe_score_tile_24.addr + (((0) / 64) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_19 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_18.stride, .addr = score_rope_h0_18.addr + (((0) / 64) * score_rope_h0_18.stride.c+(0) * score_rope_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_18.stride.c+(0) * score_rope_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_19.addr, q_rope_h0_13.addr, pe_score_tile_26.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_24.addr, score_latent_h0_18.addr, score_rope_h0_18.addr, &score_tile_24.shape, (score_tile_24.default_stride ? NULL : &score_tile_24.stride), (score_latent_h0_18.default_stride ? NULL : &score_latent_h0_18.stride), (score_rope_h0_18.default_stride ? NULL : &score_rope_h0_18.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_24.addr, score_tile_24.addr, mask_row_12.addr, &score_tile_24.shape, (score_tile_24.default_stride ? NULL : &score_tile_24.stride), (score_tile_24.default_stride ? NULL : &score_tile_24.stride), (mask_row_12.default_stride ? NULL : &mask_row_12.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_24.addr, score_tile_24.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_24.shape, (score_tile_24.default_stride ? NULL : &score_tile_24.stride), (score_tile_24.default_stride ? NULL : &score_tile_24.stride), DT_FP32);
    {
    scalar_t scores_max_12_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_12.addr, scores_max_12_scalar_f32, &scores_max_12.shape, (scores_max_12.default_stride ? NULL : &scores_max_12.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_12 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_12.stride, 0, &tmp_buffer_max_12.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_24.shape.n, score_tile_24.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_24.shape.n, score_tile_24.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_24.shape.n, score_tile_24.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_24.shape.n, score_tile_24.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_24.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, score_tile_24.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_24.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_24.shape.w) {
          dim4 fill_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, align_w - score_tile_24.shape.w};
          int elem_size = 4;
          int offset = score_tile_24.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_24.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_24.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_24.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_12_stride;
    tpu_aligned_stride(&scores_max_12_stride, 0, &scores_max_12.shape, DT_FP32);
    scores_max_12_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_24.addr, score_tile_24.addr, scores_max_12.addr, &score_tile_24.shape, (score_tile_24.default_stride ? NULL : &score_tile_24.stride), (score_tile_24.default_stride ? NULL : &score_tile_24.stride), &scores_max_12_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_12.addr);
    tpu_bdc_load_fp32_exp_table(table_12.addr);
    tpu_bdc_fp32_exp(score_tile_24.addr, score_tile_24.addr, work_score0_12.addr, work_score1_12.addr, coeff_12.addr, table_12.addr, &score_tile_24.shape);
    __ppl_tensor_info tmp_buffer_sum_12 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_12.stride, 0, &tmp_buffer_sum_12.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_24.shape.n, score_tile_24.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_24.shape.n, score_tile_24.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_24.shape.n, score_tile_24.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_24.shape.n, score_tile_24.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_24.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, score_tile_24.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_24.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_24.shape.w) {
          dim4 fill_shape = {score_tile_24.shape.n, score_tile_24.shape.c, 1, align_w - score_tile_24.shape.w};
          int elem_size = 4;
          int offset = score_tile_24.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_24.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_24.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_24.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_12.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_25 = {.shape = {1, 1, 1, 192} , .stride = score_tile_24.stride, .addr = score_tile_24.addr + ((0) * score_tile_24.stride.c+(0) * score_tile_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_24.stride.c+(0) * score_tile_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_24.default_stride};
    __ppl_tensor_info prob_tile_25 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_24.stride, .addr = prob_tile_24.addr + ((0) * prob_tile_24.stride.c+(0) * prob_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_24.stride.c+(0) * prob_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_24.default_stride};
    tpu_bdc_cast(prob_tile_25.addr, score_tile_25.addr, &prob_tile_25.shape, (prob_tile_25.default_stride ? NULL : &prob_tile_25.stride), (score_tile_25.default_stride ? NULL : &score_tile_25.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_24_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_24.addr, ctx_acc_24_scalar_f32, &ctx_acc_24.shape, (ctx_acc_24.default_stride ? NULL : &ctx_acc_24.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_24.addr, prob_tile_24.addr, kv_cache_full_24.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_12_stride;
    tpu_aligned_stride(&scores_sum_12_stride, 0, &scores_sum_12.shape, DT_FP32);
    scores_sum_12_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_24.addr, ctx_acc_24.addr, scores_sum_12.addr, &ctx_acc_24.shape, (ctx_acc_24.default_stride ? NULL : &ctx_acc_24.stride), (ctx_acc_24.default_stride ? NULL : &ctx_acc_24.stride), &scores_sum_12_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_25 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_24.stride, .addr = ctx_acc_24.addr + ((0) * ctx_acc_24.stride.c+(0) * ctx_acc_24.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_24.stride.c+(0) * ctx_acc_24.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_24.default_stride};
    __ppl_tensor_info ctx_head_37 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_36.stride, .addr = ctx_head_36.addr + ((0) * ctx_head_36.stride.c+(0) * ctx_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_36.stride.c+(0) * ctx_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_36.default_stride};
    tpu_bdc_cast(ctx_head_37.addr, ctx_acc_25.addr, &ctx_head_37.shape, (ctx_head_37.default_stride ? NULL : &ctx_head_37.stride), (ctx_acc_25.default_stride ? NULL : &ctx_acc_25.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_36_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_36.addr, out_part_36_scalar_f32, &out_part_36.shape, (out_part_36.default_stride ? NULL : &out_part_36.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_38 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_36.stride, .addr = ctx_head_36.addr + (((0) / 64) * ctx_head_36.stride.c+(0) * ctx_head_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_36.stride.c+(0) * ctx_head_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_39 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_36.stride, .addr = wukv_full_tile_36.addr + (((128) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_37 = {.shape = {1, 1, 1, 128}, .stride = out_part_36.stride, .addr = out_part_36.addr + (((0) / 64) * out_part_36.stride.c+(0) * out_part_36.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_36.stride.c+(0) * out_part_36.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_37.addr, ctx_head_38.addr, wukv_full_tile_39.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_38 = {.shape = {1, 1, 1, 128} , .stride = out_part_36.stride, .addr = out_part_36.addr + ((0) * out_part_36.stride.c+(0) * out_part_36.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_36.stride.c+(0) * out_part_36.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_36.default_stride};
    __ppl_tensor_info out_cast_37 = {.shape = {1, 1, 1, 128} , .stride = out_cast_36.stride, .addr = out_cast_36.addr + ((0) * out_cast_36.stride.c+(0) * out_cast_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_36.stride.c+(0) * out_cast_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_36.default_stride};
    tpu_bdc_cast(out_cast_37.addr, out_part_38.addr, &out_cast_37.shape, (out_cast_37.default_stride ? NULL : &out_cast_37.stride), (out_part_38.default_stride ? NULL : &out_part_38.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_38 = {.shape = {1, 1, 1, 128} , .stride = out_cast_36.stride, .addr = out_cast_36.addr + ((0) * out_cast_36.stride.c+(0) * out_cast_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_36.stride.c+(0) * out_cast_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_36.default_stride};
    __ppl_tensor_info OUT_12 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1536) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1536) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_12.addr, out_cast_38.addr, &OUT_12.shape, (OUT_12.default_stride ? NULL : &OUT_12.stride), (out_cast_38.default_stride ? NULL : &out_cast_38.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_39 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_39.stride, 0, &q_nope_head_39.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_39 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_39.stride, 0, &q_wukv_acc_39.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_39 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_39.stride, 0, &q_wukv_head_39.shape, DT_BFP16);
    __ppl_tensor_info score_tile_26 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_26.stride, 0, &score_tile_26.shape, DT_FP32);
    __ppl_tensor_info prob_tile_26 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_26.stride, 0, &prob_tile_26.shape, DT_BFP16);
    __ppl_tensor_info scores_max_13 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_13.stride, 0, &scores_max_13.shape, DT_FP32);
    __ppl_tensor_info scores_sum_13 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_13.stride, 0, &scores_sum_13.shape, DT_FP32);
    __ppl_tensor_info work_score0_13 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_13.stride, 0, &work_score0_13.shape, DT_FP32);
    __ppl_tensor_info work_score1_13 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_13.stride, 0, &work_score1_13.shape, DT_FP32);
    __ppl_tensor_info coeff_13 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_13.stride, 0, &coeff_13.shape, DT_FP32);
    __ppl_tensor_info table_13 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_13.stride, 0, &table_13.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_26 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_26.stride, 0, &ctx_acc_26.shape, DT_FP32);
    __ppl_tensor_info ctx_head_39 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_39.stride, 0, &ctx_head_39.shape, DT_BFP16);
    __ppl_tensor_info out_part_39 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_39.stride, 0, &out_part_39.shape, DT_FP32);
    __ppl_tensor_info out_cast_39 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_39.stride, 0, &out_cast_39.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_27 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_24.stride, .addr = q_upper_h1_24.addr + ((0) * q_upper_h1_24.stride.c+(0) * q_upper_h1_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_24.stride.c+(0) * q_upper_h1_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_24.default_stride};
    __ppl_tensor_info q_nope_head_40 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_39.stride, .addr = q_nope_head_39.addr + ((0) * q_nope_head_39.stride.c+(0) * q_nope_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_39.stride.c+(0) * q_nope_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_39.default_stride};
    tpu_bdc_cpy(q_nope_head_40.addr, q_upper_h1_27.addr, &q_nope_head_40.shape, (q_nope_head_40.default_stride ? NULL : &q_nope_head_40.stride), (q_upper_h1_27.default_stride ? NULL : &q_upper_h1_27.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_39_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_39.addr, q_wukv_acc_39_scalar_f32, &q_wukv_acc_39.shape, (q_wukv_acc_39.default_stride ? NULL : &q_wukv_acc_39.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_41 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_39.stride, .addr = q_nope_head_39.addr + (((0) / 64) * q_nope_head_39.stride.c+(0) * q_nope_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_39.stride.c+(0) * q_nope_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_40 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_36.stride, .addr = wukv_full_tile_36.addr + (((256) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_40 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_39.stride, .addr = q_wukv_acc_39.addr + (((0) / 64) * q_wukv_acc_39.stride.c+(0) * q_wukv_acc_39.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_39.stride.c+(0) * q_wukv_acc_39.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_40.addr, q_nope_head_41.addr, wukv_full_tile_40.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_41 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_39.stride, .addr = q_wukv_acc_39.addr + ((0) * q_wukv_acc_39.stride.c+(0) * q_wukv_acc_39.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_39.stride.c+(0) * q_wukv_acc_39.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_39.default_stride};
    __ppl_tensor_info q_wukv_head_40 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_39.stride, .addr = q_wukv_head_39.addr + ((0) * q_wukv_head_39.stride.c+(0) * q_wukv_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_39.stride.c+(0) * q_wukv_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_39.default_stride};
    tpu_bdc_cast(q_wukv_head_40.addr, q_wukv_acc_41.addr, &q_wukv_head_40.shape, (q_wukv_head_40.default_stride ? NULL : &q_wukv_head_40.stride), (q_wukv_acc_41.default_stride ? NULL : &q_wukv_acc_41.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_41 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_39.stride, .addr = q_wukv_head_39.addr + (((0) / 64) * q_wukv_head_39.stride.c+(0) * q_wukv_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_39.stride.c+(0) * q_wukv_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_27 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_24.stride, .addr = kv_cache_full_24.addr + (((0) / 64) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_24.stride.c+(0) * kv_cache_full_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_20 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_18.stride, .addr = score_latent_h0_18.addr + (((0) / 64) * score_latent_h0_18.stride.c+(0) * score_latent_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_18.stride.c+(0) * score_latent_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_20.addr, q_wukv_head_41.addr, kv_cache_full_27.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_13 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_12.stride, .addr = q_rope_h1_12.addr + (((0) / 64) * q_rope_h1_12.stride.c+(0) * q_rope_h1_12.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_12.stride.c+(0) * q_rope_h1_12.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_27 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_24.stride, .addr = pe_score_tile_24.addr + (((0) / 64) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_24.stride.c+(0) * pe_score_tile_24.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_20 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_18.stride, .addr = score_rope_h0_18.addr + (((0) / 64) * score_rope_h0_18.stride.c+(0) * score_rope_h0_18.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_18.stride.c+(0) * score_rope_h0_18.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_20.addr, q_rope_h1_13.addr, pe_score_tile_27.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_26.addr, score_latent_h0_18.addr, score_rope_h0_18.addr, &score_tile_26.shape, (score_tile_26.default_stride ? NULL : &score_tile_26.stride), (score_latent_h0_18.default_stride ? NULL : &score_latent_h0_18.stride), (score_rope_h0_18.default_stride ? NULL : &score_rope_h0_18.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_26.addr, score_tile_26.addr, mask_row_12.addr, &score_tile_26.shape, (score_tile_26.default_stride ? NULL : &score_tile_26.stride), (score_tile_26.default_stride ? NULL : &score_tile_26.stride), (mask_row_12.default_stride ? NULL : &mask_row_12.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_26.addr, score_tile_26.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_26.shape, (score_tile_26.default_stride ? NULL : &score_tile_26.stride), (score_tile_26.default_stride ? NULL : &score_tile_26.stride), DT_FP32);
    {
    scalar_t scores_max_13_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_13.addr, scores_max_13_scalar_f32, &scores_max_13.shape, (scores_max_13.default_stride ? NULL : &scores_max_13.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_13 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_13.stride, 0, &tmp_buffer_max_13.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_26.shape.n, score_tile_26.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_26.shape.n, score_tile_26.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_26.shape.n, score_tile_26.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_26.shape.n, score_tile_26.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_26.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, score_tile_26.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_26.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_26.shape.w) {
          dim4 fill_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, align_w - score_tile_26.shape.w};
          int elem_size = 4;
          int offset = score_tile_26.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_26.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_26.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_26.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_13_stride;
    tpu_aligned_stride(&scores_max_13_stride, 0, &scores_max_13.shape, DT_FP32);
    scores_max_13_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_26.addr, score_tile_26.addr, scores_max_13.addr, &score_tile_26.shape, (score_tile_26.default_stride ? NULL : &score_tile_26.stride), (score_tile_26.default_stride ? NULL : &score_tile_26.stride), &scores_max_13_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_13.addr);
    tpu_bdc_load_fp32_exp_table(table_13.addr);
    tpu_bdc_fp32_exp(score_tile_26.addr, score_tile_26.addr, work_score0_13.addr, work_score1_13.addr, coeff_13.addr, table_13.addr, &score_tile_26.shape);
    __ppl_tensor_info tmp_buffer_sum_13 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_13.stride, 0, &tmp_buffer_sum_13.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_26.shape.n, score_tile_26.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_26.shape.n, score_tile_26.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_26.shape.n, score_tile_26.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_26.shape.n, score_tile_26.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_26.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, score_tile_26.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_26.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_26.shape.w) {
          dim4 fill_shape = {score_tile_26.shape.n, score_tile_26.shape.c, 1, align_w - score_tile_26.shape.w};
          int elem_size = 4;
          int offset = score_tile_26.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_26.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_26.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_26.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_13.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_27 = {.shape = {1, 1, 1, 192} , .stride = score_tile_26.stride, .addr = score_tile_26.addr + ((0) * score_tile_26.stride.c+(0) * score_tile_26.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_26.stride.c+(0) * score_tile_26.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_26.default_stride};
    __ppl_tensor_info prob_tile_27 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_26.stride, .addr = prob_tile_26.addr + ((0) * prob_tile_26.stride.c+(0) * prob_tile_26.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_26.stride.c+(0) * prob_tile_26.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_26.default_stride};
    tpu_bdc_cast(prob_tile_27.addr, score_tile_27.addr, &prob_tile_27.shape, (prob_tile_27.default_stride ? NULL : &prob_tile_27.stride), (score_tile_27.default_stride ? NULL : &score_tile_27.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_26_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_26.addr, ctx_acc_26_scalar_f32, &ctx_acc_26.shape, (ctx_acc_26.default_stride ? NULL : &ctx_acc_26.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_26.addr, prob_tile_26.addr, kv_cache_full_24.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_13_stride;
    tpu_aligned_stride(&scores_sum_13_stride, 0, &scores_sum_13.shape, DT_FP32);
    scores_sum_13_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_26.addr, ctx_acc_26.addr, scores_sum_13.addr, &ctx_acc_26.shape, (ctx_acc_26.default_stride ? NULL : &ctx_acc_26.stride), (ctx_acc_26.default_stride ? NULL : &ctx_acc_26.stride), &scores_sum_13_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_27 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_26.stride, .addr = ctx_acc_26.addr + ((0) * ctx_acc_26.stride.c+(0) * ctx_acc_26.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_26.stride.c+(0) * ctx_acc_26.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_26.default_stride};
    __ppl_tensor_info ctx_head_40 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_39.stride, .addr = ctx_head_39.addr + ((0) * ctx_head_39.stride.c+(0) * ctx_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_39.stride.c+(0) * ctx_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_39.default_stride};
    tpu_bdc_cast(ctx_head_40.addr, ctx_acc_27.addr, &ctx_head_40.shape, (ctx_head_40.default_stride ? NULL : &ctx_head_40.stride), (ctx_acc_27.default_stride ? NULL : &ctx_acc_27.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_39_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_39.addr, out_part_39_scalar_f32, &out_part_39.shape, (out_part_39.default_stride ? NULL : &out_part_39.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_41 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_39.stride, .addr = ctx_head_39.addr + (((0) / 64) * ctx_head_39.stride.c+(0) * ctx_head_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_39.stride.c+(0) * ctx_head_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_41 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_36.stride, .addr = wukv_full_tile_36.addr + (((384) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_36.stride.c+(0) * wukv_full_tile_36.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_40 = {.shape = {1, 1, 1, 128}, .stride = out_part_39.stride, .addr = out_part_39.addr + (((0) / 64) * out_part_39.stride.c+(0) * out_part_39.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_39.stride.c+(0) * out_part_39.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_40.addr, ctx_head_41.addr, wukv_full_tile_41.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_41 = {.shape = {1, 1, 1, 128} , .stride = out_part_39.stride, .addr = out_part_39.addr + ((0) * out_part_39.stride.c+(0) * out_part_39.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_39.stride.c+(0) * out_part_39.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_39.default_stride};
    __ppl_tensor_info out_cast_40 = {.shape = {1, 1, 1, 128} , .stride = out_cast_39.stride, .addr = out_cast_39.addr + ((0) * out_cast_39.stride.c+(0) * out_cast_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_39.stride.c+(0) * out_cast_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_39.default_stride};
    tpu_bdc_cast(out_cast_40.addr, out_part_41.addr, &out_cast_40.shape, (out_cast_40.default_stride ? NULL : &out_cast_40.stride), (out_part_41.default_stride ? NULL : &out_part_41.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_41 = {.shape = {1, 1, 1, 128} , .stride = out_cast_39.stride, .addr = out_cast_39.addr + ((0) * out_cast_39.stride.c+(0) * out_cast_39.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_39.stride.c+(0) * out_cast_39.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_39.default_stride};
    __ppl_tensor_info OUT_13 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1664) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1664) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_13.addr, out_cast_41.addr, &OUT_13.shape, (OUT_13.default_stride ? NULL : &OUT_13.stride), (out_cast_41.default_stride ? NULL : &out_cast_41.stride), DT_BFP16);
  }
  __ppl_tensor_info q_upper_acc_h0_21 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h0_21.stride, 0, &q_upper_acc_h0_21.shape, DT_FP32);
  __ppl_tensor_info q_upper_h0_28 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h0_28.stride, 0, &q_upper_h0_28.shape, DT_BFP16);
  __ppl_tensor_info q_upper_acc_h1_21 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 33536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_acc_h1_21.stride, 0, &q_upper_acc_h1_21.shape, DT_FP32);
  __ppl_tensor_info q_upper_h1_28 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 14848, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_upper_h1_28.stride, 0, &q_upper_h1_28.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h0_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h0_14.stride, 0, &q_rope_src_h0_14.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h0_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h0_14.stride, 0, &q_rope_h0_14.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h0_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h0_7.stride, 0, &q_rope_tmp0_h0_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h0_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h0_7.stride, 0, &q_rope_tmp1_h0_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h0_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h0_7.stride, 0, &q_rope_tmp2_h0_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h0_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h0_7.stride, 0, &q_rope_tmp3_h0_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src_h1_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_h1_14.stride, 0, &q_rope_src_h1_14.shape, DT_BFP16);
  __ppl_tensor_info q_rope_h1_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_h1_14.stride, 0, &q_rope_h1_14.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp0_h1_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp0_h1_7.stride, 0, &q_rope_tmp0_h1_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp1_h1_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17280, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp1_h1_7.stride, 0, &q_rope_tmp1_h1_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp2_h1_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp2_h1_7.stride, 0, &q_rope_tmp2_h1_7.shape, DT_BFP16);
  __ppl_tensor_info q_rope_tmp3_h1_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_tmp3_h1_7.stride, 0, &q_rope_tmp3_h1_7.shape, DT_BFP16);
  __ppl_tensor_info cos_shared_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8320, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_shared_14.stride, 0, &cos_shared_14.shape, DT_BFP16);
  __ppl_tensor_info sin_shared_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8192, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_shared_14.stride, 0, &sin_shared_14.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_14.stride, 0, &pe_cur_14.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_cos_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_cos_7.stride, 0, &pe_cur_cos_7.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_sin_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_sin_7.stride, 0, &pe_cur_sin_7.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_7.stride, 0, &pe_cur_neg_7.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_neg_sin_7 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_neg_sin_7.stride, 0, &pe_cur_neg_sin_7.shape, DT_BFP16);
  __ppl_tensor_info pe_cur_rope_14 = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 8576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_cur_rope_14.stride, 0, &pe_cur_rope_14.shape, DT_BFP16);
  __ppl_tensor_info wukv_full_tile_42 = {.shape = { 1, 512, 1, 512}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 8192, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&wukv_full_tile_42.stride, 0, &wukv_full_tile_42.shape, DT_BFP16);
  __ppl_tensor_info kv_cache_full_28 = {.shape = { 1, 192, 1, 512}, .stride = {0}, .addr = 8960, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 3072, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&kv_cache_full_28.stride, 0, &kv_cache_full_28.shape, DT_BFP16);
  __ppl_tensor_info pe_score_tile_28 = {.shape = { 1, 192, 1, 64}, .stride = {0}, .addr = 17920, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&pe_score_tile_28.stride, 0, &pe_score_tile_28.shape, DT_BFP16);
  __ppl_tensor_info mask_row_14 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 20352, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&mask_row_14.stride, 0, &mask_row_14.shape, DT_FP32);
  __ppl_tensor_info score_latent_h0_21 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h0_21.stride, 0, &score_latent_h0_21.shape, DT_FP32);
  __ppl_tensor_info score_rope_h0_21 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h0_21.stride, 0, &score_rope_h0_21.shape, DT_FP32);
  __ppl_tensor_info score_latent_h1_7 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_latent_h1_7.stride, 0, &score_latent_h1_7.shape, DT_FP32);
  __ppl_tensor_info score_rope_h1_7 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&score_rope_h1_7.stride, 0, &score_rope_h1_7.shape, DT_FP32);
  if (tpu_workitem_index() == 7) {
    __ppl_tensor_info COS_7 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_shared_15 = {.shape = {1, 1, 1, 64} , .stride = cos_shared_14.stride, .addr = cos_shared_14.addr + ((0) * cos_shared_14.stride.c+(0) * cos_shared_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_shared_14.stride.c+(0) * cos_shared_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_shared_14.default_stride};
    tpu_gdma_cpy_S2L(cos_shared_15.addr, COS_7.addr, &cos_shared_15.shape, (cos_shared_15.default_stride ? NULL : &cos_shared_15.stride), (COS_7.default_stride ? NULL : &COS_7.stride), DT_BFP16);
    __ppl_tensor_info SIN_7 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_shared_15 = {.shape = {1, 1, 1, 64} , .stride = sin_shared_14.stride, .addr = sin_shared_14.addr + ((0) * sin_shared_14.stride.c+(0) * sin_shared_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_shared_14.stride.c+(0) * sin_shared_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_shared_14.default_stride};
    tpu_gdma_cpy_S2L(sin_shared_15.addr, SIN_7.addr, &sin_shared_15.shape, (sin_shared_15.default_stride ? NULL : &sin_shared_15.stride), (SIN_7.default_stride ? NULL : &SIN_7.stride), DT_BFP16);
    for (int d_14 = 0; d_14 < 4; ++d_14) {
      __ppl_tensor_info kv_cur_tile_21 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_cur_tile_21.stride, 0, &kv_cur_tile_21.shape, DT_BFP16);
      __ppl_tensor_info KV_CUR_7 = {.shape = {1, 1, 1, 128} , .stride = {512, 512, 512, 1} , .addr = v18.addr + ((0) * 512+((d_14 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_14 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_cur_tile_22 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_21.stride, .addr = kv_cur_tile_21.addr + ((0) * kv_cur_tile_21.stride.c+(0) * kv_cur_tile_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_21.stride.c+(0) * kv_cur_tile_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_21.default_stride};
      tpu_gdma_cpy_S2L(kv_cur_tile_22.addr, KV_CUR_7.addr, &kv_cur_tile_22.shape, (kv_cur_tile_22.default_stride ? NULL : &kv_cur_tile_22.stride), (KV_CUR_7.default_stride ? NULL : &KV_CUR_7.stride), DT_BFP16);
      __ppl_tensor_info kv_cur_tile_23 = {.shape = {1, 1, 1, 128} , .stride = kv_cur_tile_21.stride, .addr = kv_cur_tile_21.addr + ((0) * kv_cur_tile_21.stride.c+(0) * kv_cur_tile_21.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cur_tile_21.stride.c+(0) * kv_cur_tile_21.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cur_tile_21.default_stride};
      __ppl_tensor_info KVcache_OUT_14 = {.shape = {1, 1, 1, 128} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((v16) * 512+((d_14 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 512+((d_14 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT_14.addr, kv_cur_tile_23.addr, &KVcache_OUT_14.shape, (KVcache_OUT_14.default_stride ? NULL : &KVcache_OUT_14.stride), (kv_cur_tile_23.default_stride ? NULL : &kv_cur_tile_23.stride), DT_BFP16);
    }
    __ppl_tensor_info PE_CUR_7 = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v19.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_cur_15 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_14.stride, .addr = pe_cur_14.addr + ((0) * pe_cur_14.stride.c+(0) * pe_cur_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_14.stride.c+(0) * pe_cur_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_14.default_stride};
    tpu_gdma_cpy_S2L(pe_cur_15.addr, PE_CUR_7.addr, &pe_cur_15.shape, (pe_cur_15.default_stride ? NULL : &pe_cur_15.stride), (PE_CUR_7.default_stride ? NULL : &PE_CUR_7.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_cos_7.addr, pe_cur_14.addr, cos_shared_14.addr, &pe_cur_cos_7.shape, (pe_cur_cos_7.default_stride ? NULL : &pe_cur_cos_7.stride), (pe_cur_14.default_stride ? NULL : &pe_cur_14.stride), (cos_shared_14.default_stride ? NULL : &cos_shared_14.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_cur_sin_7.addr, pe_cur_14.addr, sin_shared_14.addr, &pe_cur_sin_7.shape, (pe_cur_sin_7.default_stride ? NULL : &pe_cur_sin_7.stride), (pe_cur_14.default_stride ? NULL : &pe_cur_14.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    scalar_t pe_cur_neg_7_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_cur_neg_7_scalar_DT_BFP16 = tpu_cast(pe_cur_neg_7_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_cur_neg_7.addr, pe_cur_14.addr, pe_cur_neg_7_scalar_DT_BFP16, &pe_cur_neg_7.shape, (pe_cur_neg_7.default_stride ? NULL : &pe_cur_neg_7.stride), (pe_cur_14.default_stride ? NULL : &pe_cur_14.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_cur_neg_sin_7.addr, pe_cur_neg_7.addr, sin_shared_14.addr, &pe_cur_neg_sin_7.shape, (pe_cur_neg_sin_7.default_stride ? NULL : &pe_cur_neg_sin_7.stride), (pe_cur_neg_7.default_stride ? NULL : &pe_cur_neg_7.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &pe_cur_rope_14.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_cur_rope_14.shape.n, .c = pe_cur_rope_14.shape.c, .h = pe_cur_rope_14.shape.h, .w = pe_cur_rope_14.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_cur_rope_14.addr, pe_cur_cos_7.addr, pe_cur_neg_sin_7.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_cur_rope_14.addr + 2, pe_cur_cos_7.addr + 2, pe_cur_sin_7.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_cur_rope_15 = {.shape = {1, 1, 1, 64} , .stride = pe_cur_rope_14.stride, .addr = pe_cur_rope_14.addr + ((0) * pe_cur_rope_14.stride.c+(0) * pe_cur_rope_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_cur_rope_14.stride.c+(0) * pe_cur_rope_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_cur_rope_14.default_stride};
    __ppl_tensor_info PEcache_OUT_14 = {.shape = {1, 1, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((v16) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((v16) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT_14.addr, pe_cur_rope_15.addr, &PEcache_OUT_14.shape, (PEcache_OUT_14.default_stride ? NULL : &PEcache_OUT_14.stride), (pe_cur_rope_15.default_stride ? NULL : &pe_cur_rope_15.stride), DT_BFP16);
    tpu_sync_all();
    {
    scalar_t q_upper_acc_h0_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h0_21.addr, q_upper_acc_h0_21_scalar_f32, &q_upper_acc_h0_21.shape, (q_upper_acc_h0_21.default_stride ? NULL : &q_upper_acc_h0_21.stride), DT_FP32);
    }
    {
    scalar_t q_upper_acc_h1_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_upper_acc_h1_21.addr, q_upper_acc_h1_21_scalar_f32, &q_upper_acc_h1_21.shape, (q_upper_acc_h1_21.default_stride ? NULL : &q_upper_acc_h1_21.stride), DT_FP32);
    }
    for (int k_7 = 0; k_7 < 12; ++k_7) {
      __ppl_tensor_info q_tile_28 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_tile_28.stride, 0, &q_tile_28.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_21 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 34304, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_fp8_21.stride, 0, &wuq_fp8_21.shape, DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_stage_56 = {.shape = { 1, 384, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_stage_56.stride, 0, &wuq_scale_stage_56.shape, DT_BFP16);
      __ppl_tensor_info wuq_tile_28 = {.shape = { 1, 384, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1536, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_tile_28.stride, 0, &wuq_tile_28.shape, DT_BFP16);
      __ppl_tensor_info Q_7 = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v17.addr + ((0) * 1536+((k_7 * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k_7 * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_tile_29 = {.shape = {1, 1, 1, 128} , .stride = q_tile_28.stride, .addr = q_tile_28.addr + ((0) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile_28.default_stride};
      tpu_gdma_cpy_S2L(q_tile_29.addr, Q_7.addr, &q_tile_29.shape, (q_tile_29.default_stride ? NULL : &q_tile_29.stride), (Q_7.default_stride ? NULL : &Q_7.stride), DT_BFP16);
      __ppl_tensor_info WUQ_7 = {.shape = {1, 384, 1, 128} , .stride = {4718592, 1536, 1536, 1} , .addr = v20.addr + ((2688) * 1536+((k_7 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((2688) * 1536+((k_7 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_fp8_22 = {.shape = {1, 384, 1, 128} , .stride = wuq_fp8_21.stride, .addr = wuq_fp8_21.addr + ((0) * wuq_fp8_21.stride.c+(0) * wuq_fp8_21.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_fp8_21.stride.c+(0) * wuq_fp8_21.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_fp8_21.default_stride};
      tpu_gdma_cpy_S2L(wuq_fp8_22.addr, WUQ_7.addr, &wuq_fp8_22.shape, (wuq_fp8_22.default_stride ? NULL : &wuq_fp8_22.stride), (WUQ_7.default_stride ? NULL : &WUQ_7.stride), DT_FP8E4M3);
      __ppl_tensor_info wuq_scale_scalar_126 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_126.stride, 0, &wuq_scale_scalar_126.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_42 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((21) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((21) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_127 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_126.stride, .addr = wuq_scale_scalar_126.addr + ((0) * wuq_scale_scalar_126.stride.c+(0) * wuq_scale_scalar_126.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_126.stride.c+(0) * wuq_scale_scalar_126.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_126.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_127.addr, WUQ_SCALE_42.addr, &wuq_scale_scalar_127.shape, (wuq_scale_scalar_127.default_stride ? NULL : &wuq_scale_scalar_127.stride), (WUQ_SCALE_42.default_stride ? NULL : &WUQ_SCALE_42.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_128 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_126.stride, .addr = wuq_scale_scalar_126.addr + (((0) / 64) * wuq_scale_scalar_126.stride.c+(0) * wuq_scale_scalar_126.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_126.stride.c+(0) * wuq_scale_scalar_126.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_57 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((0) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_57.addr, wuq_scale_scalar_128.addr, &wuq_scale_stage_57.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_129 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_129.stride, 0, &wuq_scale_scalar_129.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_43 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((21) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((21) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_130 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_129.stride, .addr = wuq_scale_scalar_129.addr + ((0) * wuq_scale_scalar_129.stride.c+(0) * wuq_scale_scalar_129.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_129.stride.c+(0) * wuq_scale_scalar_129.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_129.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_130.addr, WUQ_SCALE_43.addr, &wuq_scale_scalar_130.shape, (wuq_scale_scalar_130.default_stride ? NULL : &wuq_scale_scalar_130.stride), (WUQ_SCALE_43.default_stride ? NULL : &WUQ_SCALE_43.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_131 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_129.stride, .addr = wuq_scale_scalar_129.addr + (((0) / 64) * wuq_scale_scalar_129.stride.c+(0) * wuq_scale_scalar_129.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_129.stride.c+(0) * wuq_scale_scalar_129.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_58 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((64) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((64) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_58.addr, wuq_scale_scalar_131.addr, &wuq_scale_stage_58.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_132 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_132.stride, 0, &wuq_scale_scalar_132.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_44 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((22) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((22) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_133 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_132.stride, .addr = wuq_scale_scalar_132.addr + ((0) * wuq_scale_scalar_132.stride.c+(0) * wuq_scale_scalar_132.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_132.stride.c+(0) * wuq_scale_scalar_132.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_132.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_133.addr, WUQ_SCALE_44.addr, &wuq_scale_scalar_133.shape, (wuq_scale_scalar_133.default_stride ? NULL : &wuq_scale_scalar_133.stride), (WUQ_SCALE_44.default_stride ? NULL : &WUQ_SCALE_44.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_134 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_132.stride, .addr = wuq_scale_scalar_132.addr + (((0) / 64) * wuq_scale_scalar_132.stride.c+(0) * wuq_scale_scalar_132.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_132.stride.c+(0) * wuq_scale_scalar_132.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_59 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((128) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_59.addr, wuq_scale_scalar_134.addr, &wuq_scale_stage_59.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_135 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_135.stride, 0, &wuq_scale_scalar_135.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_45 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((22) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((22) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_136 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_135.stride, .addr = wuq_scale_scalar_135.addr + ((0) * wuq_scale_scalar_135.stride.c+(0) * wuq_scale_scalar_135.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_135.stride.c+(0) * wuq_scale_scalar_135.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_135.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_136.addr, WUQ_SCALE_45.addr, &wuq_scale_scalar_136.shape, (wuq_scale_scalar_136.default_stride ? NULL : &wuq_scale_scalar_136.stride), (WUQ_SCALE_45.default_stride ? NULL : &WUQ_SCALE_45.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_137 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_135.stride, .addr = wuq_scale_scalar_135.addr + (((0) / 64) * wuq_scale_scalar_135.stride.c+(0) * wuq_scale_scalar_135.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_135.stride.c+(0) * wuq_scale_scalar_135.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_60 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((192) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_60.addr, wuq_scale_scalar_137.addr, &wuq_scale_stage_60.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_138 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_138.stride, 0, &wuq_scale_scalar_138.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_46 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((23) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((23) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_139 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_138.stride, .addr = wuq_scale_scalar_138.addr + ((0) * wuq_scale_scalar_138.stride.c+(0) * wuq_scale_scalar_138.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_138.stride.c+(0) * wuq_scale_scalar_138.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_138.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_139.addr, WUQ_SCALE_46.addr, &wuq_scale_scalar_139.shape, (wuq_scale_scalar_139.default_stride ? NULL : &wuq_scale_scalar_139.stride), (WUQ_SCALE_46.default_stride ? NULL : &WUQ_SCALE_46.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_140 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_138.stride, .addr = wuq_scale_scalar_138.addr + (((0) / 64) * wuq_scale_scalar_138.stride.c+(0) * wuq_scale_scalar_138.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_138.stride.c+(0) * wuq_scale_scalar_138.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_61 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((256) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_61.addr, wuq_scale_scalar_140.addr, &wuq_scale_stage_61.shape, DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_141 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wuq_scale_scalar_141.stride, 0, &wuq_scale_scalar_141.shape, DT_BFP16);
      __ppl_tensor_info WUQ_SCALE_47 = {.shape = {1, 1, 1, 1} , .stride = {288, 12, 12, 1} , .addr = v22.addr + ((23) * 12+(k_7) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((23) * 12+(k_7) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_scalar_142 = {.shape = {1, 1, 1, 1} , .stride = wuq_scale_scalar_141.stride, .addr = wuq_scale_scalar_141.addr + ((0) * wuq_scale_scalar_141.stride.c+(0) * wuq_scale_scalar_141.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_scale_scalar_141.stride.c+(0) * wuq_scale_scalar_141.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_scale_scalar_141.default_stride};
      tpu_gdma_cpy_S2L(wuq_scale_scalar_142.addr, WUQ_SCALE_47.addr, &wuq_scale_scalar_142.shape, (wuq_scale_scalar_142.default_stride ? NULL : &wuq_scale_scalar_142.stride), (WUQ_SCALE_47.default_stride ? NULL : &WUQ_SCALE_47.stride), DT_BFP16);
      __ppl_tensor_info wuq_scale_scalar_143 = {.shape = {1, 1, 1, 1}, .stride = wuq_scale_scalar_141.stride, .addr = wuq_scale_scalar_141.addr + (((0) / 64) * wuq_scale_scalar_141.stride.c+(0) * wuq_scale_scalar_141.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_scalar_141.stride.c+(0) * wuq_scale_scalar_141.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_62 = {.shape = {1, 64, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((320) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((320) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_npu_bcast(wuq_scale_stage_62.addr, wuq_scale_scalar_143.addr, &wuq_scale_stage_62.shape, DT_BFP16);
      __ppl_tensor_info wuq_fp8_23 = {.shape = {1, 384, 1, 128}, .stride = wuq_fp8_21.stride, .addr = wuq_fp8_21.addr + (((0) / 64) * wuq_fp8_21.stride.c+(0) * wuq_fp8_21.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_fp8_21.stride.c+(0) * wuq_fp8_21.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_scale_stage_63 = {.shape = {1, 384, 1, 1}, .stride = wuq_scale_stage_56.stride, .addr = wuq_scale_stage_56.addr + (((0) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_scale_stage_56.stride.c+(0) * wuq_scale_stage_56.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_29 = {.shape = {1, 384, 1, 128}, .stride = wuq_tile_28.stride, .addr = wuq_tile_28.addr + (((0) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wuq_tile_29.addr, wuq_fp8_23.addr, &wuq_tile_29.shape, (wuq_tile_29.default_stride ? NULL : &wuq_tile_29.stride), (wuq_fp8_23.default_stride ? NULL : &wuq_fp8_23.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wuq_scale_stage_63_block_stride;
      tpu_aligned_stride(&wuq_scale_stage_63_block_stride, 0, &wuq_scale_stage_63.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wuq_scale_stage_63_block_stride.w = 0;
      tpu_bdc_fp_mul(wuq_tile_29.addr, wuq_tile_29.addr, wuq_scale_stage_63.addr, &wuq_tile_29.shape, (wuq_tile_29.default_stride ? NULL : &wuq_tile_29.stride), (wuq_tile_29.default_stride ? NULL : &wuq_tile_29.stride), &wuq_scale_stage_63_block_stride, DT_BFP16);
      __ppl_tensor_info q_tile_30 = {.shape = {1, 1, 1, 128}, .stride = q_tile_28.stride, .addr = q_tile_28.addr + (((0) / 64) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_30 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_28.stride, .addr = wuq_tile_28.addr + (((0) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h0_22 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h0_21.stride, .addr = q_upper_acc_h0_21.addr + (((0) / 64) * q_upper_acc_h0_21.stride.c+(0) * q_upper_acc_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h0_21.stride.c+(0) * q_upper_acc_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h0_22.addr, q_tile_30.addr, wuq_tile_30.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
      __ppl_tensor_info q_tile_31 = {.shape = {1, 1, 1, 128}, .stride = q_tile_28.stride, .addr = q_tile_28.addr + (((0) / 64) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_tile_28.stride.c+(0) * q_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wuq_tile_31 = {.shape = {1, 192, 1, 128}, .stride = wuq_tile_28.stride, .addr = wuq_tile_28.addr + (((192) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((192) / 64) * wuq_tile_28.stride.c+(0) * wuq_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info q_upper_acc_h1_22 = {.shape = {1, 1, 1, 192}, .stride = q_upper_acc_h1_21.stride, .addr = q_upper_acc_h1_21.addr + (((0) / 64) * q_upper_acc_h1_21.stride.c+(0) * q_upper_acc_h1_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_upper_acc_h1_21.stride.c+(0) * q_upper_acc_h1_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_fp_mm_R_trans_with_bias(q_upper_acc_h1_22.addr, q_tile_31.addr, wuq_tile_31.addr, 1, 128, 192, DT_FP32, DT_BFP16, DT_FP32, true, true, (var_context_t){0}, false);
    }
    __ppl_tensor_info q_upper_acc_h0_23 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h0_21.stride, .addr = q_upper_acc_h0_21.addr + ((0) * q_upper_acc_h0_21.stride.c+(0) * q_upper_acc_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h0_21.stride.c+(0) * q_upper_acc_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h0_21.default_stride};
    __ppl_tensor_info q_upper_h0_29 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h0_28.stride, .addr = q_upper_h0_28.addr + ((0) * q_upper_h0_28.stride.c+(0) * q_upper_h0_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_28.stride.c+(0) * q_upper_h0_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_28.default_stride};
    tpu_bdc_cast(q_upper_h0_29.addr, q_upper_acc_h0_23.addr, &q_upper_h0_29.shape, (q_upper_h0_29.default_stride ? NULL : &q_upper_h0_29.stride), (q_upper_acc_h0_23.default_stride ? NULL : &q_upper_acc_h0_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_upper_acc_h1_23 = {.shape = {1, 1, 1, 192} , .stride = q_upper_acc_h1_21.stride, .addr = q_upper_acc_h1_21.addr + ((0) * q_upper_acc_h1_21.stride.c+(0) * q_upper_acc_h1_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_upper_acc_h1_21.stride.c+(0) * q_upper_acc_h1_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_upper_acc_h1_21.default_stride};
    __ppl_tensor_info q_upper_h1_29 = {.shape = {1, 1, 1, 192} , .stride = q_upper_h1_28.stride, .addr = q_upper_h1_28.addr + ((0) * q_upper_h1_28.stride.c+(0) * q_upper_h1_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_28.stride.c+(0) * q_upper_h1_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_28.default_stride};
    tpu_bdc_cast(q_upper_h1_29.addr, q_upper_acc_h1_23.addr, &q_upper_h1_29.shape, (q_upper_h1_29.default_stride ? NULL : &q_upper_h1_29.stride), (q_upper_acc_h1_23.default_stride ? NULL : &q_upper_acc_h1_23.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int d_15 = 0; d_15 < 4; ++d_15) {
      __ppl_tensor_info wukv_fp8_21 = {.shape = { 1, 512, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_fp8_21.stride, 0, &wukv_fp8_21.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_stage_42 = {.shape = { 1, 512, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_stage_42.stride, 0, &wukv_scale_stage_42.shape, DT_BFP16);
      __ppl_tensor_info WUKV_7 = {.shape = {1, 512, 1, 128} , .stride = {2097152, 512, 512, 1} , .addr = v21.addr + ((3584) * 512+((d_15 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((3584) * 512+((d_15 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_fp8_22 = {.shape = {1, 512, 1, 128} , .stride = wukv_fp8_21.stride, .addr = wukv_fp8_21.addr + ((0) * wukv_fp8_21.stride.c+(0) * wukv_fp8_21.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_fp8_21.stride.c+(0) * wukv_fp8_21.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_fp8_21.default_stride};
      tpu_gdma_cpy_S2L(wukv_fp8_22.addr, WUKV_7.addr, &wukv_fp8_22.shape, (wukv_fp8_22.default_stride ? NULL : &wukv_fp8_22.stride), (WUKV_7.default_stride ? NULL : &WUKV_7.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_scale_scalar_70 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_70.stride, 0, &wukv_scale_scalar_70.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_28 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((28) * 4+(d_15) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((28) * 4+(d_15) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_71 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_70.stride, .addr = wukv_scale_scalar_70.addr + ((0) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_70.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_71.addr, WUKV_SCALE_28.addr, &wukv_scale_scalar_71.shape, (wukv_scale_scalar_71.default_stride ? NULL : &wukv_scale_scalar_71.stride), (WUKV_SCALE_28.default_stride ? NULL : &WUKV_SCALE_28.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_72 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_70.stride, .addr = wukv_scale_scalar_70.addr + (((0) / 64) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_43 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_42.stride, .addr = wukv_scale_stage_42.addr + (((0) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_43_chunk = wukv_scale_stage_43;
      wukv_scale_stage_43_chunk.shape.c = 64;
      wukv_scale_stage_43_chunk.addr = wukv_scale_stage_43.addr + ((0) * wukv_scale_stage_43.stride.c) * 2;
      wukv_scale_stage_43_chunk.offset = wukv_scale_stage_43.offset + ((0) * wukv_scale_stage_43.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_43_chunk.addr, wukv_scale_scalar_72.addr, &wukv_scale_stage_43_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_43_chunk_1 = wukv_scale_stage_43;
      wukv_scale_stage_43_chunk_1.shape.c = 64;
      wukv_scale_stage_43_chunk_1.addr = wukv_scale_stage_43.addr + ((1) * wukv_scale_stage_43.stride.c) * 2;
      wukv_scale_stage_43_chunk_1.offset = wukv_scale_stage_43.offset + ((1) * wukv_scale_stage_43.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_43_chunk_1.addr, wukv_scale_scalar_72.addr, &wukv_scale_stage_43_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_29 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((29) * 4+(d_15) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((29) * 4+(d_15) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_73 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_70.stride, .addr = wukv_scale_scalar_70.addr + ((0) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_70.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_73.addr, WUKV_SCALE_29.addr, &wukv_scale_scalar_73.shape, (wukv_scale_scalar_73.default_stride ? NULL : &wukv_scale_scalar_73.stride), (WUKV_SCALE_29.default_stride ? NULL : &WUKV_SCALE_29.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_74 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_70.stride, .addr = wukv_scale_scalar_70.addr + (((0) / 64) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_70.stride.c+(0) * wukv_scale_scalar_70.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_44 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_42.stride, .addr = wukv_scale_stage_42.addr + (((128) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_44_chunk = wukv_scale_stage_44;
      wukv_scale_stage_44_chunk.shape.c = 64;
      wukv_scale_stage_44_chunk.addr = wukv_scale_stage_44.addr + ((0) * wukv_scale_stage_44.stride.c) * 2;
      wukv_scale_stage_44_chunk.offset = wukv_scale_stage_44.offset + ((0) * wukv_scale_stage_44.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_44_chunk.addr, wukv_scale_scalar_74.addr, &wukv_scale_stage_44_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_44_chunk_1 = wukv_scale_stage_44;
      wukv_scale_stage_44_chunk_1.shape.c = 64;
      wukv_scale_stage_44_chunk_1.addr = wukv_scale_stage_44.addr + ((1) * wukv_scale_stage_44.stride.c) * 2;
      wukv_scale_stage_44_chunk_1.offset = wukv_scale_stage_44.offset + ((1) * wukv_scale_stage_44.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_44_chunk_1.addr, wukv_scale_scalar_74.addr, &wukv_scale_stage_44_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_75 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 8448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_scale_scalar_75.stride, 0, &wukv_scale_scalar_75.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_30 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((30) * 4+(d_15) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((30) * 4+(d_15) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_76 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_75.stride, .addr = wukv_scale_scalar_75.addr + ((0) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_75.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_76.addr, WUKV_SCALE_30.addr, &wukv_scale_scalar_76.shape, (wukv_scale_scalar_76.default_stride ? NULL : &wukv_scale_scalar_76.stride), (WUKV_SCALE_30.default_stride ? NULL : &WUKV_SCALE_30.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_77 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_75.stride, .addr = wukv_scale_scalar_75.addr + (((0) / 64) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_45 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_42.stride, .addr = wukv_scale_stage_42.addr + (((256) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_45_chunk = wukv_scale_stage_45;
      wukv_scale_stage_45_chunk.shape.c = 64;
      wukv_scale_stage_45_chunk.addr = wukv_scale_stage_45.addr + ((0) * wukv_scale_stage_45.stride.c) * 2;
      wukv_scale_stage_45_chunk.offset = wukv_scale_stage_45.offset + ((0) * wukv_scale_stage_45.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_45_chunk.addr, wukv_scale_scalar_77.addr, &wukv_scale_stage_45_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_45_chunk_1 = wukv_scale_stage_45;
      wukv_scale_stage_45_chunk_1.shape.c = 64;
      wukv_scale_stage_45_chunk_1.addr = wukv_scale_stage_45.addr + ((1) * wukv_scale_stage_45.stride.c) * 2;
      wukv_scale_stage_45_chunk_1.offset = wukv_scale_stage_45.offset + ((1) * wukv_scale_stage_45.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_45_chunk_1.addr, wukv_scale_scalar_77.addr, &wukv_scale_stage_45_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info WUKV_SCALE_31 = {.shape = {1, 1, 1, 1} , .stride = {128, 4, 4, 1} , .addr = v23.addr + ((31) * 4+(d_15) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((31) * 4+(d_15) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_scalar_78 = {.shape = {1, 1, 1, 1} , .stride = wukv_scale_scalar_75.stride, .addr = wukv_scale_scalar_75.addr + ((0) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_scale_scalar_75.default_stride};
      tpu_gdma_cpy_S2L(wukv_scale_scalar_78.addr, WUKV_SCALE_31.addr, &wukv_scale_scalar_78.shape, (wukv_scale_scalar_78.default_stride ? NULL : &wukv_scale_scalar_78.stride), (WUKV_SCALE_31.default_stride ? NULL : &WUKV_SCALE_31.stride), DT_BFP16);
      __ppl_tensor_info wukv_scale_scalar_79 = {.shape = {1, 1, 1, 1}, .stride = wukv_scale_scalar_75.stride, .addr = wukv_scale_scalar_75.addr + (((0) / 64) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_scalar_75.stride.c+(0) * wukv_scale_scalar_75.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_46 = {.shape = {1, 128, 1, 1}, .stride = wukv_scale_stage_42.stride, .addr = wukv_scale_stage_42.addr + (((384) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_46_chunk = wukv_scale_stage_46;
      wukv_scale_stage_46_chunk.shape.c = 64;
      wukv_scale_stage_46_chunk.addr = wukv_scale_stage_46.addr + ((0) * wukv_scale_stage_46.stride.c) * 2;
      wukv_scale_stage_46_chunk.offset = wukv_scale_stage_46.offset + ((0) * wukv_scale_stage_46.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_46_chunk.addr, wukv_scale_scalar_79.addr, &wukv_scale_stage_46_chunk.shape, DT_BFP16);
      // Align with Sophgo MLA PPL NPU-lane broadcast: split C dimension by 64 rows.
      __ppl_tensor_info wukv_scale_stage_46_chunk_1 = wukv_scale_stage_46;
      wukv_scale_stage_46_chunk_1.shape.c = 64;
      wukv_scale_stage_46_chunk_1.addr = wukv_scale_stage_46.addr + ((1) * wukv_scale_stage_46.stride.c) * 2;
      wukv_scale_stage_46_chunk_1.offset = wukv_scale_stage_46.offset + ((1) * wukv_scale_stage_46.stride.c) * 2;
      tpu_bdc_npu_bcast(wukv_scale_stage_46_chunk_1.addr, wukv_scale_scalar_79.addr, &wukv_scale_stage_46_chunk_1.shape, DT_BFP16);
      __ppl_tensor_info wukv_fp8_23 = {.shape = {1, 512, 1, 128}, .stride = wukv_fp8_21.stride, .addr = wukv_fp8_21.addr + (((0) / 64) * wukv_fp8_21.stride.c+(0) * wukv_fp8_21.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_fp8_21.stride.c+(0) * wukv_fp8_21.stride.w ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_scale_stage_47 = {.shape = {1, 512, 1, 1}, .stride = wukv_scale_stage_42.stride, .addr = wukv_scale_stage_42.addr + (((0) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_scale_stage_42.stride.c+(0) * wukv_scale_stage_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_full_tile_43 = {.shape = {1, 512, 1, 128}, .stride = wukv_full_tile_42.stride, .addr = wukv_full_tile_42.addr + (((0) / 64) * wukv_full_tile_42.stride.c+((d_15 * 128)) * wukv_full_tile_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_42.stride.c+((d_15 * 128)) * wukv_full_tile_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_bdc_cast(wukv_full_tile_43.addr, wukv_fp8_23.addr, &wukv_full_tile_43.shape, (wukv_full_tile_43.default_stride ? NULL : &wukv_full_tile_43.stride), (wukv_fp8_23.default_stride ? NULL : &wukv_fp8_23.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      dim4 wukv_scale_stage_47_block_stride;
      tpu_aligned_stride(&wukv_scale_stage_47_block_stride, 0, &wukv_scale_stage_47.shape, DT_BFP16);
      // Align with Sophgo MLA PPL compact scale: keep C stride, broadcast only W.
      wukv_scale_stage_47_block_stride.w = 0;
      tpu_bdc_fp_mul(wukv_full_tile_43.addr, wukv_full_tile_43.addr, wukv_scale_stage_47.addr, &wukv_full_tile_43.shape, (wukv_full_tile_43.default_stride ? NULL : &wukv_full_tile_43.stride), (wukv_full_tile_43.default_stride ? NULL : &wukv_full_tile_43.stride), &wukv_scale_stage_47_block_stride, DT_BFP16);
    }
    __ppl_tensor_info KVcache_OUT_15 = {.shape = {1, 192, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v30.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_29 = {.shape = {1, 192, 1, 512} , .stride = kv_cache_full_28.stride, .addr = kv_cache_full_28.addr + ((0) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_cache_full_28.default_stride};
    tpu_gdma_cpy_S2L(kv_cache_full_29.addr, KVcache_OUT_15.addr, &kv_cache_full_29.shape, (kv_cache_full_29.default_stride ? NULL : &kv_cache_full_29.stride), (KVcache_OUT_15.default_stride ? NULL : &KVcache_OUT_15.stride), DT_BFP16);
    __ppl_tensor_info PEcache_OUT_15 = {.shape = {1, 192, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v31.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_29 = {.shape = {1, 192, 1, 64} , .stride = pe_score_tile_28.stride, .addr = pe_score_tile_28.addr + ((0) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile_28.default_stride};
    tpu_gdma_cpy_S2L(pe_score_tile_29.addr, PEcache_OUT_15.addr, &pe_score_tile_29.shape, (pe_score_tile_29.default_stride ? NULL : &pe_score_tile_29.stride), (PEcache_OUT_15.default_stride ? NULL : &PEcache_OUT_15.stride), DT_BFP16);
    __ppl_tensor_info DECODE_MASK_7 = {.shape = {1, 1, 1, 192} , .stride = {192, 192, 192, 1} , .addr = v28.addr + ((0) * 192+(0) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+(0) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_row_15 = {.shape = {1, 1, 1, 192} , .stride = mask_row_14.stride, .addr = mask_row_14.addr + ((0) * mask_row_14.stride.c+(0) * mask_row_14.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_row_14.stride.c+(0) * mask_row_14.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_row_14.default_stride};
    tpu_gdma_cpy_S2L(mask_row_15.addr, DECODE_MASK_7.addr, &mask_row_15.shape, (mask_row_15.default_stride ? NULL : &mask_row_15.stride), (DECODE_MASK_7.default_stride ? NULL : &DECODE_MASK_7.stride), DT_FP32);
    {
    scalar_t score_latent_h0_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h0_21.addr, score_latent_h0_21_scalar_f32, &score_latent_h0_21.shape, (score_latent_h0_21.default_stride ? NULL : &score_latent_h0_21.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h0_21_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h0_21.addr, score_rope_h0_21_scalar_f32, &score_rope_h0_21.shape, (score_rope_h0_21.default_stride ? NULL : &score_rope_h0_21.stride), DT_FP32);
    }
    {
    scalar_t score_latent_h1_7_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_h1_7.addr, score_latent_h1_7_scalar_f32, &score_latent_h1_7.shape, (score_latent_h1_7.default_stride ? NULL : &score_latent_h1_7.stride), DT_FP32);
    }
    {
    scalar_t score_rope_h1_7_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_h1_7.addr, score_rope_h1_7_scalar_f32, &score_rope_h1_7.shape, (score_rope_h1_7.default_stride ? NULL : &score_rope_h1_7.stride), DT_FP32);
    }
    __ppl_tensor_info q_upper_h0_30 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h0_28.stride, .addr = q_upper_h0_28.addr + ((0) * q_upper_h0_28.stride.c+(128) * q_upper_h0_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_28.stride.c+(128) * q_upper_h0_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_28.default_stride};
    __ppl_tensor_info q_rope_src_h0_15 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h0_14.stride, .addr = q_rope_src_h0_14.addr + ((0) * q_rope_src_h0_14.stride.c+(0) * q_rope_src_h0_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h0_14.stride.c+(0) * q_rope_src_h0_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h0_14.default_stride};
    tpu_bdc_cpy(q_rope_src_h0_15.addr, q_upper_h0_30.addr, &q_rope_src_h0_15.shape, (q_rope_src_h0_15.default_stride ? NULL : &q_rope_src_h0_15.stride), (q_upper_h0_30.default_stride ? NULL : &q_upper_h0_30.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h0_7.addr, q_rope_src_h0_14.addr, cos_shared_14.addr, &q_rope_tmp0_h0_7.shape, (q_rope_tmp0_h0_7.default_stride ? NULL : &q_rope_tmp0_h0_7.stride), (q_rope_src_h0_14.default_stride ? NULL : &q_rope_src_h0_14.stride), (cos_shared_14.default_stride ? NULL : &cos_shared_14.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h0_7.addr, q_rope_src_h0_14.addr, sin_shared_14.addr, &q_rope_tmp1_h0_7.shape, (q_rope_tmp1_h0_7.default_stride ? NULL : &q_rope_tmp1_h0_7.stride), (q_rope_src_h0_14.default_stride ? NULL : &q_rope_src_h0_14.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h0_7_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h0_7_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h0_7_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h0_7.addr, q_rope_src_h0_14.addr, q_rope_tmp2_h0_7_scalar_DT_BFP16, &q_rope_tmp2_h0_7.shape, (q_rope_tmp2_h0_7.default_stride ? NULL : &q_rope_tmp2_h0_7.stride), (q_rope_src_h0_14.default_stride ? NULL : &q_rope_src_h0_14.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h0_7.addr, q_rope_tmp2_h0_7.addr, sin_shared_14.addr, &q_rope_tmp3_h0_7.shape, (q_rope_tmp3_h0_7.default_stride ? NULL : &q_rope_tmp3_h0_7.stride), (q_rope_tmp2_h0_7.default_stride ? NULL : &q_rope_tmp2_h0_7.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h0_14.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h0_14.shape.n, .c = q_rope_h0_14.shape.c, .h = q_rope_h0_14.shape.h, .w = q_rope_h0_14.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h0_14.addr, q_rope_tmp0_h0_7.addr, q_rope_tmp3_h0_7.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h0_14.addr + 2, q_rope_tmp0_h0_7.addr + 2, q_rope_tmp1_h0_7.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_upper_h1_30 = {.shape = {1, 1, 1, 64} , .stride = q_upper_h1_28.stride, .addr = q_upper_h1_28.addr + ((0) * q_upper_h1_28.stride.c+(128) * q_upper_h1_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_28.stride.c+(128) * q_upper_h1_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_28.default_stride};
    __ppl_tensor_info q_rope_src_h1_15 = {.shape = {1, 1, 1, 64} , .stride = q_rope_src_h1_14.stride, .addr = q_rope_src_h1_14.addr + ((0) * q_rope_src_h1_14.stride.c+(0) * q_rope_src_h1_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_h1_14.stride.c+(0) * q_rope_src_h1_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src_h1_14.default_stride};
    tpu_bdc_cpy(q_rope_src_h1_15.addr, q_upper_h1_30.addr, &q_rope_src_h1_15.shape, (q_rope_src_h1_15.default_stride ? NULL : &q_rope_src_h1_15.stride), (q_upper_h1_30.default_stride ? NULL : &q_upper_h1_30.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp0_h1_7.addr, q_rope_src_h1_14.addr, cos_shared_14.addr, &q_rope_tmp0_h1_7.shape, (q_rope_tmp0_h1_7.default_stride ? NULL : &q_rope_tmp0_h1_7.stride), (q_rope_src_h1_14.default_stride ? NULL : &q_rope_src_h1_14.stride), (cos_shared_14.default_stride ? NULL : &cos_shared_14.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_rope_tmp1_h1_7.addr, q_rope_src_h1_14.addr, sin_shared_14.addr, &q_rope_tmp1_h1_7.shape, (q_rope_tmp1_h1_7.default_stride ? NULL : &q_rope_tmp1_h1_7.stride), (q_rope_src_h1_14.default_stride ? NULL : &q_rope_src_h1_14.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    scalar_t q_rope_tmp2_h1_7_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_rope_tmp2_h1_7_scalar_DT_BFP16 = tpu_cast(q_rope_tmp2_h1_7_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_rope_tmp2_h1_7.addr, q_rope_src_h1_14.addr, q_rope_tmp2_h1_7_scalar_DT_BFP16, &q_rope_tmp2_h1_7.shape, (q_rope_tmp2_h1_7.default_stride ? NULL : &q_rope_tmp2_h1_7.stride), (q_rope_src_h1_14.default_stride ? NULL : &q_rope_src_h1_14.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_rope_tmp3_h1_7.addr, q_rope_tmp2_h1_7.addr, sin_shared_14.addr, &q_rope_tmp3_h1_7.shape, (q_rope_tmp3_h1_7.default_stride ? NULL : &q_rope_tmp3_h1_7.stride), (q_rope_tmp2_h1_7.default_stride ? NULL : &q_rope_tmp2_h1_7.stride), (sin_shared_14.default_stride ? NULL : &sin_shared_14.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_rope_h1_14.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_rope_h1_14.shape.n, .c = q_rope_h1_14.shape.c, .h = q_rope_h1_14.shape.h, .w = q_rope_h1_14.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_rope_h1_14.addr, q_rope_tmp0_h1_7.addr, q_rope_tmp3_h1_7.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_rope_h1_14.addr + 2, q_rope_tmp0_h1_7.addr + 2, q_rope_tmp1_h1_7.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info q_nope_head_42 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_42.stride, 0, &q_nope_head_42.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_42 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_42.stride, 0, &q_wukv_acc_42.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_42 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_42.stride, 0, &q_wukv_head_42.shape, DT_BFP16);
    __ppl_tensor_info score_tile_28 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_28.stride, 0, &score_tile_28.shape, DT_FP32);
    __ppl_tensor_info prob_tile_28 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_28.stride, 0, &prob_tile_28.shape, DT_BFP16);
    __ppl_tensor_info scores_max_14 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_14.stride, 0, &scores_max_14.shape, DT_FP32);
    __ppl_tensor_info scores_sum_14 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_14.stride, 0, &scores_sum_14.shape, DT_FP32);
    __ppl_tensor_info work_score0_14 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_14.stride, 0, &work_score0_14.shape, DT_FP32);
    __ppl_tensor_info work_score1_14 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_14.stride, 0, &work_score1_14.shape, DT_FP32);
    __ppl_tensor_info coeff_14 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_14.stride, 0, &coeff_14.shape, DT_FP32);
    __ppl_tensor_info table_14 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_14.stride, 0, &table_14.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_28 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_28.stride, 0, &ctx_acc_28.shape, DT_FP32);
    __ppl_tensor_info ctx_head_42 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_42.stride, 0, &ctx_head_42.shape, DT_BFP16);
    __ppl_tensor_info out_part_42 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_42.stride, 0, &out_part_42.shape, DT_FP32);
    __ppl_tensor_info out_cast_42 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_42.stride, 0, &out_cast_42.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h0_31 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h0_28.stride, .addr = q_upper_h0_28.addr + ((0) * q_upper_h0_28.stride.c+(0) * q_upper_h0_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h0_28.stride.c+(0) * q_upper_h0_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h0_28.default_stride};
    __ppl_tensor_info q_nope_head_43 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_42.stride, .addr = q_nope_head_42.addr + ((0) * q_nope_head_42.stride.c+(0) * q_nope_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_42.stride.c+(0) * q_nope_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_42.default_stride};
    tpu_bdc_cpy(q_nope_head_43.addr, q_upper_h0_31.addr, &q_nope_head_43.shape, (q_nope_head_43.default_stride ? NULL : &q_nope_head_43.stride), (q_upper_h0_31.default_stride ? NULL : &q_upper_h0_31.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_42_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_42.addr, q_wukv_acc_42_scalar_f32, &q_wukv_acc_42.shape, (q_wukv_acc_42.default_stride ? NULL : &q_wukv_acc_42.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_44 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_42.stride, .addr = q_nope_head_42.addr + (((0) / 64) * q_nope_head_42.stride.c+(0) * q_nope_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_42.stride.c+(0) * q_nope_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_44 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_42.stride, .addr = wukv_full_tile_42.addr + (((0) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_43 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_42.stride, .addr = q_wukv_acc_42.addr + (((0) / 64) * q_wukv_acc_42.stride.c+(0) * q_wukv_acc_42.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_42.stride.c+(0) * q_wukv_acc_42.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_43.addr, q_nope_head_44.addr, wukv_full_tile_44.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_44 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_42.stride, .addr = q_wukv_acc_42.addr + ((0) * q_wukv_acc_42.stride.c+(0) * q_wukv_acc_42.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_42.stride.c+(0) * q_wukv_acc_42.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_42.default_stride};
    __ppl_tensor_info q_wukv_head_43 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_42.stride, .addr = q_wukv_head_42.addr + ((0) * q_wukv_head_42.stride.c+(0) * q_wukv_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_42.stride.c+(0) * q_wukv_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_42.default_stride};
    tpu_bdc_cast(q_wukv_head_43.addr, q_wukv_acc_44.addr, &q_wukv_head_43.shape, (q_wukv_head_43.default_stride ? NULL : &q_wukv_head_43.stride), (q_wukv_acc_44.default_stride ? NULL : &q_wukv_acc_44.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_44 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_42.stride, .addr = q_wukv_head_42.addr + (((0) / 64) * q_wukv_head_42.stride.c+(0) * q_wukv_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_42.stride.c+(0) * q_wukv_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_30 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_28.stride, .addr = kv_cache_full_28.addr + (((0) / 64) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_22 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_21.stride, .addr = score_latent_h0_21.addr + (((0) / 64) * score_latent_h0_21.stride.c+(0) * score_latent_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_21.stride.c+(0) * score_latent_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_22.addr, q_wukv_head_44.addr, kv_cache_full_30.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h0_15 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h0_14.stride, .addr = q_rope_h0_14.addr + (((0) / 64) * q_rope_h0_14.stride.c+(0) * q_rope_h0_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h0_14.stride.c+(0) * q_rope_h0_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_30 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_28.stride, .addr = pe_score_tile_28.addr + (((0) / 64) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_22 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_21.stride, .addr = score_rope_h0_21.addr + (((0) / 64) * score_rope_h0_21.stride.c+(0) * score_rope_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_21.stride.c+(0) * score_rope_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_22.addr, q_rope_h0_15.addr, pe_score_tile_30.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_28.addr, score_latent_h0_21.addr, score_rope_h0_21.addr, &score_tile_28.shape, (score_tile_28.default_stride ? NULL : &score_tile_28.stride), (score_latent_h0_21.default_stride ? NULL : &score_latent_h0_21.stride), (score_rope_h0_21.default_stride ? NULL : &score_rope_h0_21.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_28.addr, score_tile_28.addr, mask_row_14.addr, &score_tile_28.shape, (score_tile_28.default_stride ? NULL : &score_tile_28.stride), (score_tile_28.default_stride ? NULL : &score_tile_28.stride), (mask_row_14.default_stride ? NULL : &mask_row_14.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_28.addr, score_tile_28.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_28.shape, (score_tile_28.default_stride ? NULL : &score_tile_28.stride), (score_tile_28.default_stride ? NULL : &score_tile_28.stride), DT_FP32);
    {
    scalar_t scores_max_14_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_14.addr, scores_max_14_scalar_f32, &scores_max_14.shape, (scores_max_14.default_stride ? NULL : &scores_max_14.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_14 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_14.stride, 0, &tmp_buffer_max_14.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_28.shape.n, score_tile_28.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_28.shape.n, score_tile_28.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_28.shape.n, score_tile_28.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_28.shape.n, score_tile_28.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_28.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, score_tile_28.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_28.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_28.shape.w) {
          dim4 fill_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, align_w - score_tile_28.shape.w};
          int elem_size = 4;
          int offset = score_tile_28.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_28.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_28.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_28.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_14_stride;
    tpu_aligned_stride(&scores_max_14_stride, 0, &scores_max_14.shape, DT_FP32);
    scores_max_14_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_28.addr, score_tile_28.addr, scores_max_14.addr, &score_tile_28.shape, (score_tile_28.default_stride ? NULL : &score_tile_28.stride), (score_tile_28.default_stride ? NULL : &score_tile_28.stride), &scores_max_14_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_14.addr);
    tpu_bdc_load_fp32_exp_table(table_14.addr);
    tpu_bdc_fp32_exp(score_tile_28.addr, score_tile_28.addr, work_score0_14.addr, work_score1_14.addr, coeff_14.addr, table_14.addr, &score_tile_28.shape);
    __ppl_tensor_info tmp_buffer_sum_14 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_14.stride, 0, &tmp_buffer_sum_14.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_28.shape.n, score_tile_28.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_28.shape.n, score_tile_28.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_28.shape.n, score_tile_28.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_28.shape.n, score_tile_28.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_28.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, score_tile_28.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_28.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_28.shape.w) {
          dim4 fill_shape = {score_tile_28.shape.n, score_tile_28.shape.c, 1, align_w - score_tile_28.shape.w};
          int elem_size = 4;
          int offset = score_tile_28.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_28.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_28.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_28.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_14.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_29 = {.shape = {1, 1, 1, 192} , .stride = score_tile_28.stride, .addr = score_tile_28.addr + ((0) * score_tile_28.stride.c+(0) * score_tile_28.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_28.stride.c+(0) * score_tile_28.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_28.default_stride};
    __ppl_tensor_info prob_tile_29 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_28.stride, .addr = prob_tile_28.addr + ((0) * prob_tile_28.stride.c+(0) * prob_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_28.stride.c+(0) * prob_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_28.default_stride};
    tpu_bdc_cast(prob_tile_29.addr, score_tile_29.addr, &prob_tile_29.shape, (prob_tile_29.default_stride ? NULL : &prob_tile_29.stride), (score_tile_29.default_stride ? NULL : &score_tile_29.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_28_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_28.addr, ctx_acc_28_scalar_f32, &ctx_acc_28.shape, (ctx_acc_28.default_stride ? NULL : &ctx_acc_28.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_28.addr, prob_tile_28.addr, kv_cache_full_28.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_14_stride;
    tpu_aligned_stride(&scores_sum_14_stride, 0, &scores_sum_14.shape, DT_FP32);
    scores_sum_14_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_28.addr, ctx_acc_28.addr, scores_sum_14.addr, &ctx_acc_28.shape, (ctx_acc_28.default_stride ? NULL : &ctx_acc_28.stride), (ctx_acc_28.default_stride ? NULL : &ctx_acc_28.stride), &scores_sum_14_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_29 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_28.stride, .addr = ctx_acc_28.addr + ((0) * ctx_acc_28.stride.c+(0) * ctx_acc_28.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_28.stride.c+(0) * ctx_acc_28.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_28.default_stride};
    __ppl_tensor_info ctx_head_43 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_42.stride, .addr = ctx_head_42.addr + ((0) * ctx_head_42.stride.c+(0) * ctx_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_42.stride.c+(0) * ctx_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_42.default_stride};
    tpu_bdc_cast(ctx_head_43.addr, ctx_acc_29.addr, &ctx_head_43.shape, (ctx_head_43.default_stride ? NULL : &ctx_head_43.stride), (ctx_acc_29.default_stride ? NULL : &ctx_acc_29.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_42_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_42.addr, out_part_42_scalar_f32, &out_part_42.shape, (out_part_42.default_stride ? NULL : &out_part_42.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_44 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_42.stride, .addr = ctx_head_42.addr + (((0) / 64) * ctx_head_42.stride.c+(0) * ctx_head_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_42.stride.c+(0) * ctx_head_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_45 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_42.stride, .addr = wukv_full_tile_42.addr + (((128) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((128) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_43 = {.shape = {1, 1, 1, 128}, .stride = out_part_42.stride, .addr = out_part_42.addr + (((0) / 64) * out_part_42.stride.c+(0) * out_part_42.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_42.stride.c+(0) * out_part_42.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_43.addr, ctx_head_44.addr, wukv_full_tile_45.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_44 = {.shape = {1, 1, 1, 128} , .stride = out_part_42.stride, .addr = out_part_42.addr + ((0) * out_part_42.stride.c+(0) * out_part_42.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_42.stride.c+(0) * out_part_42.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_42.default_stride};
    __ppl_tensor_info out_cast_43 = {.shape = {1, 1, 1, 128} , .stride = out_cast_42.stride, .addr = out_cast_42.addr + ((0) * out_cast_42.stride.c+(0) * out_cast_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_42.stride.c+(0) * out_cast_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_42.default_stride};
    tpu_bdc_cast(out_cast_43.addr, out_part_44.addr, &out_cast_43.shape, (out_cast_43.default_stride ? NULL : &out_cast_43.stride), (out_part_44.default_stride ? NULL : &out_part_44.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_44 = {.shape = {1, 1, 1, 128} , .stride = out_cast_42.stride, .addr = out_cast_42.addr + ((0) * out_cast_42.stride.c+(0) * out_cast_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_42.stride.c+(0) * out_cast_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_42.default_stride};
    __ppl_tensor_info OUT_14 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1792) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1792) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_14.addr, out_cast_44.addr, &OUT_14.shape, (OUT_14.default_stride ? NULL : &OUT_14.stride), (out_cast_44.default_stride ? NULL : &out_cast_44.stride), DT_BFP16);
    __ppl_tensor_info q_nope_head_45 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_head_45.stride, 0, &q_nope_head_45.shape, DT_BFP16);
    __ppl_tensor_info q_wukv_acc_45 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 18304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_acc_45.stride, 0, &q_wukv_acc_45.shape, DT_FP32);
    __ppl_tensor_info q_wukv_head_45 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_wukv_head_45.stride, 0, &q_wukv_head_45.shape, DT_BFP16);
    __ppl_tensor_info score_tile_30 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 8192, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_30.stride, 0, &score_tile_30.shape, DT_FP32);
    __ppl_tensor_info prob_tile_30 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 384, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile_30.stride, 0, &prob_tile_30.shape, DT_BFP16);
    __ppl_tensor_info scores_max_15 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_15.stride, 0, &scores_max_15.shape, DT_FP32);
    __ppl_tensor_info scores_sum_15 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 16768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum_15.stride, 0, &scores_sum_15.shape, DT_FP32);
    __ppl_tensor_info work_score0_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0_15.stride, 0, &work_score0_15.shape, DT_FP32);
    __ppl_tensor_info work_score1_15 = {.shape = { 1, 1, 1, 192}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1_15.stride, 0, &work_score1_15.shape, DT_FP32);
    __ppl_tensor_info coeff_15 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff_15.stride, 0, &coeff_15.shape, DT_FP32);
    __ppl_tensor_info table_15 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table_15.stride, 0, &table_15.shape, DT_FP32);
    __ppl_tensor_info ctx_acc_30 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 12032, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_acc_30.stride, 0, &ctx_acc_30.shape, DT_FP32);
    __ppl_tensor_info ctx_head_45 = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&ctx_head_45.stride, 0, &ctx_head_45.shape, DT_BFP16);
    __ppl_tensor_info out_part_45 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part_45.stride, 0, &out_part_45.shape, DT_FP32);
    __ppl_tensor_info out_cast_45 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_45.stride, 0, &out_cast_45.shape, DT_BFP16);
    __ppl_tensor_info q_upper_h1_31 = {.shape = {1, 1, 1, 128} , .stride = q_upper_h1_28.stride, .addr = q_upper_h1_28.addr + ((0) * q_upper_h1_28.stride.c+(0) * q_upper_h1_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_upper_h1_28.stride.c+(0) * q_upper_h1_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_upper_h1_28.default_stride};
    __ppl_tensor_info q_nope_head_46 = {.shape = {1, 1, 1, 128} , .stride = q_nope_head_45.stride, .addr = q_nope_head_45.addr + ((0) * q_nope_head_45.stride.c+(0) * q_nope_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope_head_45.stride.c+(0) * q_nope_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope_head_45.default_stride};
    tpu_bdc_cpy(q_nope_head_46.addr, q_upper_h1_31.addr, &q_nope_head_46.shape, (q_nope_head_46.default_stride ? NULL : &q_nope_head_46.stride), (q_upper_h1_31.default_stride ? NULL : &q_upper_h1_31.stride), DT_BFP16);
    {
    scalar_t q_wukv_acc_45_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_wukv_acc_45.addr, q_wukv_acc_45_scalar_f32, &q_wukv_acc_45.shape, (q_wukv_acc_45.default_stride ? NULL : &q_wukv_acc_45.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_head_47 = {.shape = {1, 1, 1, 128}, .stride = q_nope_head_45.stride, .addr = q_nope_head_45.addr + (((0) / 64) * q_nope_head_45.stride.c+(0) * q_nope_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_nope_head_45.stride.c+(0) * q_nope_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_46 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_42.stride, .addr = wukv_full_tile_42.addr + (((256) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((256) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_wukv_acc_46 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_acc_45.stride, .addr = q_wukv_acc_45.addr + (((0) / 64) * q_wukv_acc_45.stride.c+(0) * q_wukv_acc_45.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_acc_45.stride.c+(0) * q_wukv_acc_45.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm(q_wukv_acc_46.addr, q_nope_head_47.addr, wukv_full_tile_46.addr, 1, 128, 512, DT_FP32, DT_BFP16, true);
    __ppl_tensor_info q_wukv_acc_47 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_acc_45.stride, .addr = q_wukv_acc_45.addr + ((0) * q_wukv_acc_45.stride.c+(0) * q_wukv_acc_45.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_acc_45.stride.c+(0) * q_wukv_acc_45.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_acc_45.default_stride};
    __ppl_tensor_info q_wukv_head_46 = {.shape = {1, 1, 1, 512} , .stride = q_wukv_head_45.stride, .addr = q_wukv_head_45.addr + ((0) * q_wukv_head_45.stride.c+(0) * q_wukv_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_head_45.stride.c+(0) * q_wukv_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_head_45.default_stride};
    tpu_bdc_cast(q_wukv_head_46.addr, q_wukv_acc_47.addr, &q_wukv_head_46.shape, (q_wukv_head_46.default_stride ? NULL : &q_wukv_head_46.stride), (q_wukv_acc_47.default_stride ? NULL : &q_wukv_acc_47.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_wukv_head_47 = {.shape = {1, 1, 1, 512}, .stride = q_wukv_head_45.stride, .addr = q_wukv_head_45.addr + (((0) / 64) * q_wukv_head_45.stride.c+(0) * q_wukv_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_wukv_head_45.stride.c+(0) * q_wukv_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_cache_full_31 = {.shape = {1, 192, 1, 512}, .stride = kv_cache_full_28.stride, .addr = kv_cache_full_28.addr + (((0) / 64) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * kv_cache_full_28.stride.c+(0) * kv_cache_full_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_latent_h0_23 = {.shape = {1, 1, 1, 192}, .stride = score_latent_h0_21.stride, .addr = score_latent_h0_21.addr + (((0) / 64) * score_latent_h0_21.stride.c+(0) * score_latent_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_latent_h0_21.stride.c+(0) * score_latent_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_latent_h0_23.addr, q_wukv_head_47.addr, kv_cache_full_31.addr, 1, 512, 192, DT_FP32, DT_BFP16);
    __ppl_tensor_info q_rope_h1_15 = {.shape = {1, 1, 1, 64}, .stride = q_rope_h1_14.stride, .addr = q_rope_h1_14.addr + (((0) / 64) * q_rope_h1_14.stride.c+(0) * q_rope_h1_14.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * q_rope_h1_14.stride.c+(0) * q_rope_h1_14.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_score_tile_31 = {.shape = {1, 192, 1, 64}, .stride = pe_score_tile_28.stride, .addr = pe_score_tile_28.addr + (((0) / 64) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * pe_score_tile_28.stride.c+(0) * pe_score_tile_28.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info score_rope_h0_23 = {.shape = {1, 1, 1, 192}, .stride = score_rope_h0_21.stride, .addr = score_rope_h0_21.addr + (((0) / 64) * score_rope_h0_21.stride.c+(0) * score_rope_h0_21.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * score_rope_h0_21.stride.c+(0) * score_rope_h0_21.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(score_rope_h0_23.addr, q_rope_h1_15.addr, pe_score_tile_31.addr, 1, 64, 192, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile_30.addr, score_latent_h0_21.addr, score_rope_h0_21.addr, &score_tile_30.shape, (score_tile_30.default_stride ? NULL : &score_tile_30.stride), (score_latent_h0_21.default_stride ? NULL : &score_latent_h0_21.stride), (score_rope_h0_21.default_stride ? NULL : &score_rope_h0_21.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile_30.addr, score_tile_30.addr, mask_row_14.addr, &score_tile_30.shape, (score_tile_30.default_stride ? NULL : &score_tile_30.stride), (score_tile_30.default_stride ? NULL : &score_tile_30.stride), (mask_row_14.default_stride ? NULL : &mask_row_14.stride), DT_FP32);
    tpu_bdc_fp_mul_C( score_tile_30.addr, score_tile_30.addr, (scalar_t){.f32 = 0.0721688}, &score_tile_30.shape, (score_tile_30.default_stride ? NULL : &score_tile_30.stride), (score_tile_30.default_stride ? NULL : &score_tile_30.stride), DT_FP32);
    {
    scalar_t scores_max_15_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max_15.addr, scores_max_15_scalar_f32, &scores_max_15.shape, (scores_max_15.default_stride ? NULL : &scores_max_15.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_15 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_15.stride, 0, &tmp_buffer_max_15.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_30.shape.n, score_tile_30.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_30.shape.n, score_tile_30.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_30.shape.n, score_tile_30.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_30.shape.n, score_tile_30.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_30.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, score_tile_30.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_30.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_30.shape.w) {
          dim4 fill_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, align_w - score_tile_30.shape.w};
          int elem_size = 4;
          int offset = score_tile_30.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_30.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_30.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_30.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    dim4 scores_max_15_stride;
    tpu_aligned_stride(&scores_max_15_stride, 0, &scores_max_15.shape, DT_FP32);
    scores_max_15_stride.w = 0;
    tpu_bdc_fp_sub( score_tile_30.addr, score_tile_30.addr, scores_max_15.addr, &score_tile_30.shape, (score_tile_30.default_stride ? NULL : &score_tile_30.stride), (score_tile_30.default_stride ? NULL : &score_tile_30.stride), &scores_max_15_stride, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff_15.addr);
    tpu_bdc_load_fp32_exp_table(table_15.addr);
    tpu_bdc_fp32_exp(score_tile_30.addr, score_tile_30.addr, work_score0_15.addr, work_score1_15.addr, coeff_15.addr, table_15.addr, &score_tile_30.shape);
    __ppl_tensor_info tmp_buffer_sum_15 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_15.stride, 0, &tmp_buffer_sum_15.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 192;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile_30.shape.n, score_tile_30.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_30.shape.n, score_tile_30.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_30.shape.n, score_tile_30.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_30.shape.n, score_tile_30.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile_30.shape.w && align_w == eu_num) {
        dim4 padded_stride = {192, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, score_tile_30.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_30.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile_30.shape.w) {
          dim4 fill_shape = {score_tile_30.shape.n, score_tile_30.shape.c, 1, align_w - score_tile_30.shape.w};
          int elem_size = 4;
          int offset = score_tile_30.shape.w * elem_size;
          dim4 fill_tensor_stride = {192, align_w, score_tile_30.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_30.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_30.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_15.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_tile_31 = {.shape = {1, 1, 1, 192} , .stride = score_tile_30.stride, .addr = score_tile_30.addr + ((0) * score_tile_30.stride.c+(0) * score_tile_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_30.stride.c+(0) * score_tile_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_30.default_stride};
    __ppl_tensor_info prob_tile_31 = {.shape = {1, 1, 1, 192} , .stride = prob_tile_30.stride, .addr = prob_tile_30.addr + ((0) * prob_tile_30.stride.c+(0) * prob_tile_30.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile_30.stride.c+(0) * prob_tile_30.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile_30.default_stride};
    tpu_bdc_cast(prob_tile_31.addr, score_tile_31.addr, &prob_tile_31.shape, (prob_tile_31.default_stride ? NULL : &prob_tile_31.stride), (score_tile_31.default_stride ? NULL : &score_tile_31.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t ctx_acc_30_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(ctx_acc_30.addr, ctx_acc_30_scalar_f32, &ctx_acc_30.shape, (ctx_acc_30.default_stride ? NULL : &ctx_acc_30.stride), DT_FP32);
    }
    tpu_bdc_fp_mm(ctx_acc_30.addr, prob_tile_30.addr, kv_cache_full_28.addr, 1, 192, 512, DT_FP32, DT_BFP16, true);
    dim4 scores_sum_15_stride;
    tpu_aligned_stride(&scores_sum_15_stride, 0, &scores_sum_15.shape, DT_FP32);
    scores_sum_15_stride.w = 0;
    tpu_bdc_fp_div( ctx_acc_30.addr, ctx_acc_30.addr, scores_sum_15.addr, &ctx_acc_30.shape, (ctx_acc_30.default_stride ? NULL : &ctx_acc_30.stride), (ctx_acc_30.default_stride ? NULL : &ctx_acc_30.stride), &scores_sum_15_stride, DT_FP32);
    __ppl_tensor_info ctx_acc_31 = {.shape = {1, 1, 1, 512} , .stride = ctx_acc_30.stride, .addr = ctx_acc_30.addr + ((0) * ctx_acc_30.stride.c+(0) * ctx_acc_30.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc_30.stride.c+(0) * ctx_acc_30.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc_30.default_stride};
    __ppl_tensor_info ctx_head_46 = {.shape = {1, 1, 1, 512} , .stride = ctx_head_45.stride, .addr = ctx_head_45.addr + ((0) * ctx_head_45.stride.c+(0) * ctx_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_head_45.stride.c+(0) * ctx_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_head_45.default_stride};
    tpu_bdc_cast(ctx_head_46.addr, ctx_acc_31.addr, &ctx_head_46.shape, (ctx_head_46.default_stride ? NULL : &ctx_head_46.stride), (ctx_acc_31.default_stride ? NULL : &ctx_acc_31.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    {
    scalar_t out_part_45_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part_45.addr, out_part_45_scalar_f32, &out_part_45.shape, (out_part_45.default_stride ? NULL : &out_part_45.stride), DT_FP32);
    }
    __ppl_tensor_info ctx_head_47 = {.shape = {1, 1, 1, 512}, .stride = ctx_head_45.stride, .addr = ctx_head_45.addr + (((0) / 64) * ctx_head_45.stride.c+(0) * ctx_head_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((0) / 64) * ctx_head_45.stride.c+(0) * ctx_head_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wukv_full_tile_47 = {.shape = {1, 128, 1, 512}, .stride = wukv_full_tile_42.stride, .addr = wukv_full_tile_42.addr + (((384) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = (((384) / 64) * wukv_full_tile_42.stride.c+(0) * wukv_full_tile_42.stride.w ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info out_part_46 = {.shape = {1, 1, 1, 128}, .stride = out_part_45.stride, .addr = out_part_45.addr + (((0) / 64) * out_part_45.stride.c+(0) * out_part_45.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = (((0) / 64) * out_part_45.stride.c+(0) * out_part_45.stride.w ) * 4, .unsigned_flag = 0, .default_stride = false};
    tpu_bdc_fp_mm_R_trans(out_part_46.addr, ctx_head_47.addr, wukv_full_tile_47.addr, 1, 512, 128, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_part_47 = {.shape = {1, 1, 1, 128} , .stride = out_part_45.stride, .addr = out_part_45.addr + ((0) * out_part_45.stride.c+(0) * out_part_45.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part_45.stride.c+(0) * out_part_45.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part_45.default_stride};
    __ppl_tensor_info out_cast_46 = {.shape = {1, 1, 1, 128} , .stride = out_cast_45.stride, .addr = out_cast_45.addr + ((0) * out_cast_45.stride.c+(0) * out_cast_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_45.stride.c+(0) * out_cast_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_45.default_stride};
    tpu_bdc_cast(out_cast_46.addr, out_part_47.addr, &out_cast_46.shape, (out_cast_46.default_stride ? NULL : &out_cast_46.stride), (out_part_47.default_stride ? NULL : &out_part_47.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_47 = {.shape = {1, 1, 1, 128} , .stride = out_cast_45.stride, .addr = out_cast_45.addr + ((0) * out_cast_45.stride.c+(0) * out_cast_45.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_45.stride.c+(0) * out_cast_45.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_45.default_stride};
    __ppl_tensor_info OUT_15 = {.shape = {1, 1, 1, 128} , .stride = {2048, 2048, 2048, 1} , .addr = v29.addr + ((0) * 2048+(1920) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 2048+(1920) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT_15.addr, out_cast_47.addr, &OUT_15.shape, (OUT_15.default_stride ? NULL : &OUT_15.stride), (out_cast_47.default_stride ? NULL : &out_cast_47.stride), DT_BFP16);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
  global_addr_t v5;
  global_addr_t v6;
  global_addr_t v7;
  global_addr_t v8;
  global_addr_t v9;
  global_addr_t v10;
  global_addr_t v11;
  global_addr_t v12;
  global_addr_t v13;
  global_addr_t v14;
  global_addr_t v15;
  int v16;
} tpu_kernel_api_latent_attention_fp8_decode_kernel_args_t;
int latent_attention_fp8_decode_kernel(const void * args) {
  tpu_kernel_api_latent_attention_fp8_decode_kernel_args_t *api = (tpu_kernel_api_latent_attention_fp8_decode_kernel_args_t*)args;
  tpu_initialize();
  latent_attention_fp8_decode_kernel_inner(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5,
    api->v6,
    api->v7,
    api->v8,
    api->v9,
    api->v10,
    api->v11,
    api->v12,
    api->v13,
    api->v14,
    api->v15,
    api->v16);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(latent_attention_fp8_decode_kernel)
