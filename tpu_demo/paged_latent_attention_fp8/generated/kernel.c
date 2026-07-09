#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

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

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11, global_addr_t v12, global_addr_t v13, global_addr_t v14, global_addr_t v15, global_addr_t v16, global_addr_t v17) {
  __ppl_tensor_info v33 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v16, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v31 = {.shape = {1, 32, 1, 512} , .stride = {0}, .addr = v14, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v32 = {.shape = {1, 32, 1, 64} , .stride = {0}, .addr = v15, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {1, 1, 1, 1536} , .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 3072, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {1, 6144, 1, 1536} , .stride = {0}, .addr = v2, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 9437184, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v30 = {.shape = {1, 1, 1, 1} , .stride = {0}, .addr = v13, .dtype = DT_UINT32, .mode = 2, .align_mode = 0, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {1, 6144, 1, 12} , .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 147456, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v27 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v10, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v34 = {.shape = {1, 32, 1, 128} , .stride = {0}, .addr = v17, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1, 8192, 1, 512} , .stride = {0}, .addr = v4, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 4194304, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v28 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v11, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v23 = {.shape = {1, 32, 1, 512} , .stride = {0}, .addr = v6, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v22 = {.shape = {1, 8192, 1, 4} , .stride = {0}, .addr = v5, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 65536, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v24 = {.shape = {1, 32, 1, 64} , .stride = {0}, .addr = v7, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 4096, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v26 = {.shape = {1, 1, 1, 64} , .stride = {0}, .addr = v9, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v29 = {.shape = {1, 2, 1, 1} , .stride = {0}, .addr = v12, .dtype = DT_UINT32, .mode = 2, .align_mode = 0, .size = 8, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v25 = {.shape = {1, 1, 1, 512} , .stride = {0}, .addr = v8, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 1024, .unsigned_flag = 0, .default_stride = true};
  int __pla_block_num = tpu_core_num();
  if (__pla_block_num > 8) {
    __pla_block_num = 8;
  }
  int __pla_block_idx = tpu_core_index();
  if (__pla_block_idx >= __pla_block_num) {
    return;
  }
  int __pla_h_per_block = (32 + __pla_block_num - 1) / __pla_block_num;
  int __pla_h_begin = __pla_block_idx * __pla_h_per_block;
  int __pla_h_end = MIN(__pla_h_begin + __pla_h_per_block, 32);
  for (int by = __pla_h_begin; by < __pla_h_end; ++by) {
    __ppl_tensor_info k_src = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_src.stride, 0, &k_src.shape, DT_BFP16);
    __ppl_tensor_info k_cos_s = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33024, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_cos_s.stride, 0, &k_cos_s.shape, DT_BFP16);
    __ppl_tensor_info k_sin_s = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_sin_s.stride, 0, &k_sin_s.shape, DT_BFP16);
    __ppl_tensor_info k_x_cos = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_x_cos.stride, 0, &k_x_cos.shape, DT_BFP16);
    __ppl_tensor_info k_x_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_x_sin.stride, 0, &k_x_sin.shape, DT_BFP16);
    __ppl_tensor_info k_neg = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_neg.stride, 0, &k_neg.shape, DT_BFP16);
    __ppl_tensor_info k_neg_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_neg_sin.stride, 0, &k_neg_sin.shape, DT_BFP16);
    __ppl_tensor_info k_rotated = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 256, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&k_rotated.stride, 0, &k_rotated.shape, DT_BFP16);
    __ppl_tensor_info q_nope = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope.stride, 0, &q_nope.shape, DT_BFP16);
    __ppl_tensor_info q_rope = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_rope.stride, 0, &q_rope.shape, DT_BFP16);
    __ppl_tensor_info q_nope_acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 128, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_acc.stride, 0, &q_nope_acc.shape, DT_FP32);
    __ppl_tensor_info q_rope_acc = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 1152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_rope_acc.stride, 0, &q_rope_acc.shape, DT_FP32);
    __ppl_tensor_info q_abs = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 19200, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_abs.stride, 0, &q_abs.shape, DT_BFP16);
    __ppl_tensor_info q_x_cos = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_x_cos.stride, 0, &q_x_cos.shape, DT_BFP16);
    __ppl_tensor_info q_x_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_x_sin.stride, 0, &q_x_sin.shape, DT_BFP16);
    __ppl_tensor_info q_neg = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 16640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_neg.stride, 0, &q_neg.shape, DT_BFP16);
    __ppl_tensor_info q_neg_sin = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_neg_sin.stride, 0, &q_neg_sin.shape, DT_BFP16);
    __ppl_tensor_info q_pe = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 20288, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_pe.stride, 0, &q_pe.shape, DT_BFP16);
    __ppl_tensor_info acc_o = {.shape = { 1, 1, 1, 512}, .stride = {0}, .addr = 17152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&acc_o.stride, 0, &acc_o.shape, DT_FP32);
    __ppl_tensor_info max_v = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 20672, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&max_v.stride, 0, &max_v.shape, DT_FP32);
    __ppl_tensor_info logsum = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 65728, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&logsum.stride, 0, &logsum.shape, DT_FP32);
    __ppl_tensor_info value_acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&value_acc.stride, 0, &value_acc.shape, DT_FP32);
    __ppl_tensor_info value_out = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&value_out.stride, 0, &value_out.shape, DT_BFP16);
    __ppl_tensor_info pe_new = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v26.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info k_src_1 = {.shape = {1, 1, 1, 64} , .stride = k_src.stride, .addr = k_src.addr + ((0) * k_src.stride.c+(0) * k_src.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * k_src.stride.c+(0) * k_src.stride.w ) * 2, .unsigned_flag = 0, .default_stride = k_src.default_stride};
    tpu_gdma_cpy_S2L(k_src_1.addr, pe_new.addr, &k_src_1.shape, (k_src_1.default_stride ? NULL : &k_src_1.stride), (pe_new.default_stride ? NULL : &pe_new.stride), DT_BFP16);
    __ppl_tensor_info cos = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info k_cos_s_1 = {.shape = {1, 1, 1, 64} , .stride = k_cos_s.stride, .addr = k_cos_s.addr + ((0) * k_cos_s.stride.c+(0) * k_cos_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * k_cos_s.stride.c+(0) * k_cos_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = k_cos_s.default_stride};
    tpu_gdma_cpy_S2L(k_cos_s_1.addr, cos.addr, &k_cos_s_1.shape, (k_cos_s_1.default_stride ? NULL : &k_cos_s_1.stride), (cos.default_stride ? NULL : &cos.stride), DT_BFP16);
    __ppl_tensor_info sin = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v28.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info k_sin_s_1 = {.shape = {1, 1, 1, 64} , .stride = k_sin_s.stride, .addr = k_sin_s.addr + ((0) * k_sin_s.stride.c+(0) * k_sin_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * k_sin_s.stride.c+(0) * k_sin_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = k_sin_s.default_stride};
    tpu_gdma_cpy_S2L(k_sin_s_1.addr, sin.addr, &k_sin_s_1.shape, (k_sin_s_1.default_stride ? NULL : &k_sin_s_1.stride), (sin.default_stride ? NULL : &sin.stride), DT_BFP16);
    tpu_bdc_fp_mul( k_x_cos.addr, k_src.addr, k_cos_s.addr, &k_x_cos.shape, (k_x_cos.default_stride ? NULL : &k_x_cos.stride), (k_src.default_stride ? NULL : &k_src.stride), (k_cos_s.default_stride ? NULL : &k_cos_s.stride), DT_BFP16);
    tpu_bdc_fp_mul( k_x_sin.addr, k_src.addr, k_sin_s.addr, &k_x_sin.shape, (k_x_sin.default_stride ? NULL : &k_x_sin.stride), (k_src.default_stride ? NULL : &k_src.stride), (k_sin_s.default_stride ? NULL : &k_sin_s.stride), DT_BFP16);
    {
    scalar_t k_neg_scalar_DT_BFP16 = {.f32 = (float)-1};
    k_neg_scalar_DT_BFP16 = tpu_cast(k_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( k_neg.addr, k_src.addr, k_neg_scalar_DT_BFP16, &k_neg.shape, (k_neg.default_stride ? NULL : &k_neg.stride), (k_src.default_stride ? NULL : &k_src.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( k_neg_sin.addr, k_neg.addr, k_sin_s.addr, &k_neg_sin.shape, (k_neg_sin.default_stride ? NULL : &k_neg_sin.stride), (k_neg.default_stride ? NULL : &k_neg.stride), (k_sin_s.default_stride ? NULL : &k_sin_s.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &k_rotated.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = k_rotated.shape.n, .c = k_rotated.shape.c, .h = k_rotated.shape.h, .w = k_rotated.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( k_rotated.addr, k_x_cos.addr, k_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( k_rotated.addr + 2, k_x_cos.addr + 2, k_x_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info k_rotated_1 = {.shape = {1, 1, 1, 64} , .stride = k_rotated.stride, .addr = k_rotated.addr + ((0) * k_rotated.stride.c+(0) * k_rotated.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * k_rotated.stride.c+(0) * k_rotated.stride.w ) * 2, .unsigned_flag = 0, .default_stride = k_rotated.default_stride};
    __ppl_tensor_info pe_new_rope = {.shape = {1, 1, 1, 64} , .stride = {64, 64, 64, 1} , .addr = v33.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    if (__pla_block_idx == 0 && by == __pla_h_begin) {
      tpu_gdma_cpy_L2S(pe_new_rope.addr, k_rotated_1.addr, &pe_new_rope.shape, (pe_new_rope.default_stride ? NULL : &pe_new_rope.stride), (k_rotated_1.default_stride ? NULL : &k_rotated_1.stride), DT_BFP16);
      {
      dim4 __scatter_shape = {1, 1, v23.shape.c, v23.shape.w};
      tpu_gdma_h_scatter_S2S(v23.addr, v25.addr, v30.addr, false, &__scatter_shape, 1, NULL, NULL, NULL, DT_BFP16);
      tpu_poll();
      }
      {
      dim4 __scatter_shape = {1, 1, v24.shape.c, v24.shape.w};
      tpu_gdma_h_scatter_S2S(v24.addr, v33.addr, v30.addr, false, &__scatter_shape, 1, NULL, NULL, NULL, DT_BFP16);
      tpu_poll();
      }
      {
      dim4 __gather_block_shape = {1, 1, 2, 8192};
      tpu_gdma_h_gather_S2S(v31.addr, v23.addr, v29.addr, false, (scalar_t){.u32 = 0}, &__gather_block_shape, 2, NULL, NULL, NULL, DT_BFP16);
      }
      {
      dim4 __gather_block_shape = {1, 1, 2, 1024};
      tpu_gdma_h_gather_S2S(v32.addr, v24.addr, v29.addr, false, (scalar_t){.u32 = 0}, &__gather_block_shape, 2, NULL, NULL, NULL, DT_BFP16);
      }
    }
    if (by == __pla_h_begin) {
      tpu_sync_core();
    }
    {
    scalar_t q_nope_acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_nope_acc.addr, q_nope_acc_scalar_f32, &q_nope_acc.shape, (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), DT_FP32);
    }
    {
    scalar_t q_rope_acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_rope_acc.addr, q_rope_acc_scalar_f32, &q_rope_acc.shape, (q_rope_acc.default_stride ? NULL : &q_rope_acc.stride), DT_FP32);
    }
    for (int ib = 0; ib < 12; ++ib) {
      __ppl_tensor_info query_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 1408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&query_shared.stride, 0, &query_shared.shape, DT_BFP16);
      __ppl_tensor_info weight_nope_fp8 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_nope_fp8.stride, 0, &weight_nope_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info weight_nope_bf16 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_nope_bf16.stride, 0, &weight_nope_bf16.shape, DT_BFP16);
      __ppl_tensor_info scale_nope = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&scale_nope.stride, 0, &scale_nope.shape, DT_BFP16);
      __ppl_tensor_info weight_nope = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 33152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_nope.stride, 0, &weight_nope.shape, DT_BFP16);
      __ppl_tensor_info partial_nope = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 33664, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&partial_nope.stride, 0, &partial_nope.shape, DT_FP32);
      __ppl_tensor_info weight_rope_fp8 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_rope_fp8.stride, 0, &weight_rope_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info weight_rope_bf16 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 640, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_rope_bf16.stride, 0, &weight_rope_bf16.shape, DT_BFP16);
      __ppl_tensor_info scale_rope = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 33152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&scale_rope.stride, 0, &scale_rope.shape, DT_BFP16);
      __ppl_tensor_info weight_rope = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_rope.stride, 0, &weight_rope.shape, DT_BFP16);
      __ppl_tensor_info partial_rope = {.shape = { 1, 1, 1, 64}, .stride = {0}, .addr = 33152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&partial_rope.stride, 0, &partial_rope.shape, DT_FP32);
      __ppl_tensor_info query = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v18.addr + ((0) * 1536+((ib * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((ib * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info query_shared_1 = {.shape = {1, 1, 1, 128} , .stride = query_shared.stride, .addr = query_shared.addr + ((0) * query_shared.stride.c+(0) * query_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * query_shared.stride.c+(0) * query_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = query_shared.default_stride};
      tpu_gdma_cpy_S2L(query_shared_1.addr, query.addr, &query_shared_1.shape, (query_shared_1.default_stride ? NULL : &query_shared_1.stride), (query.default_stride ? NULL : &query.stride), DT_BFP16);
      __ppl_tensor_info wuq = {.shape = {1, 128, 1, 128} , .stride = {9437184, 1536, 1536, 1} , .addr = v19.addr + (((by * 192)) * 1536+((ib * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = (((by * 192)) * 1536+((ib * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_nope_fp8_1 = {.shape = {1, 128, 1, 128} , .stride = weight_nope_fp8.stride, .addr = weight_nope_fp8.addr + ((0) * weight_nope_fp8.stride.c+(0) * weight_nope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_nope_fp8.stride.c+(0) * weight_nope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_nope_fp8.default_stride};
      tpu_gdma_cpy_S2L(weight_nope_fp8_1.addr, wuq.addr, &weight_nope_fp8_1.shape, (weight_nope_fp8_1.default_stride ? NULL : &weight_nope_fp8_1.stride), (wuq.default_stride ? NULL : &wuq.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_nope_fp8_2 = {.shape = {1, 128, 1, 128} , .stride = weight_nope_fp8.stride, .addr = weight_nope_fp8.addr + ((0) * weight_nope_fp8.stride.c+(0) * weight_nope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_nope_fp8.stride.c+(0) * weight_nope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_nope_fp8.default_stride};
      __ppl_tensor_info weight_nope_bf16_1 = {.shape = {1, 128, 1, 128} , .stride = weight_nope_bf16.stride, .addr = weight_nope_bf16.addr + ((0) * weight_nope_bf16.stride.c+(0) * weight_nope_bf16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_nope_bf16.stride.c+(0) * weight_nope_bf16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_nope_bf16.default_stride};
      tpu_bdc_cast(weight_nope_bf16_1.addr, weight_nope_fp8_2.addr, &weight_nope_bf16_1.shape, (weight_nope_bf16_1.default_stride ? NULL : &weight_nope_bf16_1.stride), (weight_nope_fp8_2.default_stride ? NULL : &weight_nope_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info wuq_scale_expanded = {.shape = {1, 128, 1, 1} , .stride = {73728, 12, 12, 1} , .addr = v20.addr + (((by * 192)) * 12+(ib) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((by * 192)) * 12+(ib) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info scale_nope_1 = {.shape = {1, 128, 1, 1} , .stride = scale_nope.stride, .addr = scale_nope.addr + ((0) * scale_nope.stride.c+(0) * scale_nope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_nope.stride.c+(0) * scale_nope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_nope.default_stride};
      tpu_gdma_cpy_S2L(scale_nope_1.addr, wuq_scale_expanded.addr, &scale_nope_1.shape, (scale_nope_1.default_stride ? NULL : &scale_nope_1.stride), (wuq_scale_expanded.default_stride ? NULL : &wuq_scale_expanded.stride), DT_BFP16);
      dim4 scale_nope_stride;
      tpu_aligned_stride(&scale_nope_stride, 0, &scale_nope.shape, DT_BFP16);
      scale_nope_stride.w = 0;
      tpu_bdc_fp_mul( weight_nope.addr, weight_nope_bf16.addr, scale_nope.addr, &weight_nope.shape, (weight_nope.default_stride ? NULL : &weight_nope.stride), (weight_nope_bf16.default_stride ? NULL : &weight_nope_bf16.stride), &scale_nope_stride, DT_BFP16);
      {
      scalar_t partial_nope_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(partial_nope.addr, partial_nope_scalar_f32, &partial_nope.shape, (partial_nope.default_stride ? NULL : &partial_nope.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(partial_nope.addr, query_shared.addr, weight_nope.addr, 1, 128, 128, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( q_nope_acc.addr, q_nope_acc.addr, partial_nope.addr, &q_nope_acc.shape, (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), (partial_nope.default_stride ? NULL : &partial_nope.stride), DT_FP32);
      __ppl_tensor_info wuq_1 = {.shape = {1, 64, 1, 128} , .stride = {9437184, 1536, 1536, 1} , .addr = v19.addr + ((((by * 192) + 128)) * 1536+((ib * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((((by * 192) + 128)) * 1536+((ib * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_rope_fp8_1 = {.shape = {1, 64, 1, 128} , .stride = weight_rope_fp8.stride, .addr = weight_rope_fp8.addr + ((0) * weight_rope_fp8.stride.c+(0) * weight_rope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_rope_fp8.stride.c+(0) * weight_rope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_rope_fp8.default_stride};
      tpu_gdma_cpy_S2L(weight_rope_fp8_1.addr, wuq_1.addr, &weight_rope_fp8_1.shape, (weight_rope_fp8_1.default_stride ? NULL : &weight_rope_fp8_1.stride), (wuq_1.default_stride ? NULL : &wuq_1.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_rope_fp8_2 = {.shape = {1, 64, 1, 128} , .stride = weight_rope_fp8.stride, .addr = weight_rope_fp8.addr + ((0) * weight_rope_fp8.stride.c+(0) * weight_rope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_rope_fp8.stride.c+(0) * weight_rope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_rope_fp8.default_stride};
      __ppl_tensor_info weight_rope_bf16_1 = {.shape = {1, 64, 1, 128} , .stride = weight_rope_bf16.stride, .addr = weight_rope_bf16.addr + ((0) * weight_rope_bf16.stride.c+(0) * weight_rope_bf16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_rope_bf16.stride.c+(0) * weight_rope_bf16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_rope_bf16.default_stride};
      tpu_bdc_cast(weight_rope_bf16_1.addr, weight_rope_fp8_2.addr, &weight_rope_bf16_1.shape, (weight_rope_bf16_1.default_stride ? NULL : &weight_rope_bf16_1.stride), (weight_rope_fp8_2.default_stride ? NULL : &weight_rope_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info wuq_scale_expanded_1 = {.shape = {1, 64, 1, 1} , .stride = {73728, 12, 12, 1} , .addr = v20.addr + ((((by * 192) + 128)) * 12+(ib) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((((by * 192) + 128)) * 12+(ib) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info scale_rope_1 = {.shape = {1, 64, 1, 1} , .stride = scale_rope.stride, .addr = scale_rope.addr + ((0) * scale_rope.stride.c+(0) * scale_rope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_rope.stride.c+(0) * scale_rope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_rope.default_stride};
      tpu_gdma_cpy_S2L(scale_rope_1.addr, wuq_scale_expanded_1.addr, &scale_rope_1.shape, (scale_rope_1.default_stride ? NULL : &scale_rope_1.stride), (wuq_scale_expanded_1.default_stride ? NULL : &wuq_scale_expanded_1.stride), DT_BFP16);
      dim4 scale_rope_stride;
      tpu_aligned_stride(&scale_rope_stride, 0, &scale_rope.shape, DT_BFP16);
      scale_rope_stride.w = 0;
      tpu_bdc_fp_mul( weight_rope.addr, weight_rope_bf16.addr, scale_rope.addr, &weight_rope.shape, (weight_rope.default_stride ? NULL : &weight_rope.stride), (weight_rope_bf16.default_stride ? NULL : &weight_rope_bf16.stride), &scale_rope_stride, DT_BFP16);
      {
      scalar_t partial_rope_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(partial_rope.addr, partial_rope_scalar_f32, &partial_rope.shape, (partial_rope.default_stride ? NULL : &partial_rope.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(partial_rope.addr, query_shared.addr, weight_rope.addr, 1, 128, 64, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( q_rope_acc.addr, q_rope_acc.addr, partial_rope.addr, &q_rope_acc.shape, (q_rope_acc.default_stride ? NULL : &q_rope_acc.stride), (q_rope_acc.default_stride ? NULL : &q_rope_acc.stride), (partial_rope.default_stride ? NULL : &partial_rope.stride), DT_FP32);
    }
    __ppl_tensor_info q_nope_acc_1 = {.shape = {1, 1, 1, 128} , .stride = q_nope_acc.stride, .addr = q_nope_acc.addr + ((0) * q_nope_acc.stride.c+(0) * q_nope_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_nope_acc.stride.c+(0) * q_nope_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_nope_acc.default_stride};
    __ppl_tensor_info q_nope_1 = {.shape = {1, 1, 1, 128} , .stride = q_nope.stride, .addr = q_nope.addr + ((0) * q_nope.stride.c+(0) * q_nope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope.stride.c+(0) * q_nope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope.default_stride};
    tpu_bdc_cast(q_nope_1.addr, q_nope_acc_1.addr, &q_nope_1.shape, (q_nope_1.default_stride ? NULL : &q_nope_1.stride), (q_nope_acc_1.default_stride ? NULL : &q_nope_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info q_rope_acc_1 = {.shape = {1, 1, 1, 64} , .stride = q_rope_acc.stride, .addr = q_rope_acc.addr + ((0) * q_rope_acc.stride.c+(0) * q_rope_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_rope_acc.stride.c+(0) * q_rope_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_rope_acc.default_stride};
    __ppl_tensor_info q_rope_1 = {.shape = {1, 1, 1, 64} , .stride = q_rope.stride, .addr = q_rope.addr + ((0) * q_rope.stride.c+(0) * q_rope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope.stride.c+(0) * q_rope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope.default_stride};
    tpu_bdc_cast(q_rope_1.addr, q_rope_acc_1.addr, &q_rope_1.shape, (q_rope_1.default_stride ? NULL : &q_rope_1.stride), (q_rope_acc_1.default_stride ? NULL : &q_rope_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    for (int ib_1 = 0; ib_1 < 4; ++ib_1) {
      __ppl_tensor_info weight_fp8 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 33024, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_fp8.stride, 0, &weight_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info weight_bf16 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 16896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_bf16.stride, 0, &weight_bf16.shape, DT_BFP16);
      __ppl_tensor_info scale_rows = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 33024, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&scale_rows.stride, 0, &scale_rows.shape, DT_BFP16);
      __ppl_tensor_info weight_dequant = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_dequant.stride, 0, &weight_dequant.shape, DT_BFP16);
      __ppl_tensor_info acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 128, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&acc.stride, 0, &acc.shape, DT_FP32);
      __ppl_tensor_info out_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&out_shared.stride, 0, &out_shared.shape, DT_BFP16);
      __ppl_tensor_info wukv = {.shape = {1, 128, 1, 128} , .stride = {4194304, 512, 512, 1} , .addr = v21.addr + (((by * 256)) * 512+((ib_1 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = (((by * 256)) * 512+((ib_1 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_fp8_1 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      tpu_gdma_cpy_S2L(weight_fp8_1.addr, wukv.addr, &weight_fp8_1.shape, (weight_fp8_1.default_stride ? NULL : &weight_fp8_1.stride), (wukv.default_stride ? NULL : &wukv.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_fp8_2 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      __ppl_tensor_info weight_bf16_1 = {.shape = {1, 128, 1, 128} , .stride = weight_bf16.stride, .addr = weight_bf16.addr + ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_bf16.default_stride};
      tpu_bdc_cast(weight_bf16_1.addr, weight_fp8_2.addr, &weight_bf16_1.shape, (weight_bf16_1.default_stride ? NULL : &weight_bf16_1.stride), (weight_fp8_2.default_stride ? NULL : &weight_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info wukv_scale_expanded = {.shape = {1, 128, 1, 1} , .stride = {32768, 4, 4, 1} , .addr = v22.addr + (((by * 256)) * 4+(ib_1) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((by * 256)) * 4+(ib_1) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info scale_rows_1 = {.shape = {1, 128, 1, 1} , .stride = scale_rows.stride, .addr = scale_rows.addr + ((0) * scale_rows.stride.c+(0) * scale_rows.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_rows.stride.c+(0) * scale_rows.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_rows.default_stride};
      tpu_gdma_cpy_S2L(scale_rows_1.addr, wukv_scale_expanded.addr, &scale_rows_1.shape, (scale_rows_1.default_stride ? NULL : &scale_rows_1.stride), (wukv_scale_expanded.default_stride ? NULL : &wukv_scale_expanded.stride), DT_BFP16);
      dim4 scale_rows_stride;
      tpu_aligned_stride(&scale_rows_stride, 0, &scale_rows.shape, DT_BFP16);
      scale_rows_stride.w = 0;
      tpu_bdc_fp_mul( weight_dequant.addr, weight_bf16.addr, scale_rows.addr, &weight_dequant.shape, (weight_dequant.default_stride ? NULL : &weight_dequant.stride), (weight_bf16.default_stride ? NULL : &weight_bf16.stride), &scale_rows_stride, DT_BFP16);
      {
      scalar_t acc_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(acc.addr, acc_scalar_f32, &acc.shape, (acc.default_stride ? NULL : &acc.stride), DT_FP32);
      }
      tpu_bdc_fp_mm(acc.addr, q_nope.addr, weight_dequant.addr, 1, 128, 128, DT_FP32, DT_BFP16, true);
      __ppl_tensor_info acc_1 = {.shape = {1, 1, 1, 128} , .stride = acc.stride, .addr = acc.addr + ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = acc.default_stride};
      __ppl_tensor_info out_shared_1 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
      tpu_bdc_cast(out_shared_1.addr, acc_1.addr, &out_shared_1.shape, (out_shared_1.default_stride ? NULL : &out_shared_1.stride), (acc_1.default_stride ? NULL : &acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info out_shared_2 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
      __ppl_tensor_info q_abs_1 = {.shape = {1, 1, 1, 128} , .stride = q_abs.stride, .addr = q_abs.addr + ((0) * q_abs.stride.c+((ib_1 * 128)) * q_abs.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_abs.stride.c+((ib_1 * 128)) * q_abs.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_abs.default_stride};
      tpu_bdc_cpy(q_abs_1.addr, out_shared_2.addr, &q_abs_1.shape, (q_abs_1.default_stride ? NULL : &q_abs_1.stride), (out_shared_2.default_stride ? NULL : &out_shared_2.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_x_cos.addr, q_rope.addr, k_cos_s.addr, &q_x_cos.shape, (q_x_cos.default_stride ? NULL : &q_x_cos.stride), (q_rope.default_stride ? NULL : &q_rope.stride), (k_cos_s.default_stride ? NULL : &k_cos_s.stride), DT_BFP16);
    tpu_bdc_fp_mul( q_x_sin.addr, q_rope.addr, k_sin_s.addr, &q_x_sin.shape, (q_x_sin.default_stride ? NULL : &q_x_sin.stride), (q_rope.default_stride ? NULL : &q_rope.stride), (k_sin_s.default_stride ? NULL : &k_sin_s.stride), DT_BFP16);
    {
    scalar_t q_neg_scalar_DT_BFP16 = {.f32 = (float)-1};
    q_neg_scalar_DT_BFP16 = tpu_cast(q_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( q_neg.addr, q_rope.addr, q_neg_scalar_DT_BFP16, &q_neg.shape, (q_neg.default_stride ? NULL : &q_neg.stride), (q_rope.default_stride ? NULL : &q_rope.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( q_neg_sin.addr, q_neg.addr, k_sin_s.addr, &q_neg_sin.shape, (q_neg_sin.default_stride ? NULL : &q_neg_sin.stride), (q_neg.default_stride ? NULL : &q_neg.stride), (k_sin_s.default_stride ? NULL : &k_sin_s.stride), DT_BFP16);
    {
    dim4 half_stride;
    tpu_aligned_stride(&half_stride, 0, &q_pe.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = q_pe.shape.n, .c = q_pe.shape.c, .h = q_pe.shape.h, .w = q_pe.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( q_pe.addr, q_x_cos.addr, q_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( q_pe.addr + 2, q_x_cos.addr + 2, q_x_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    {
    scalar_t acc_o_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(acc_o.addr, acc_o_scalar_f32, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), DT_FP32);
    }
    {
    scalar_t logsum_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(logsum.addr, logsum_scalar_f32, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), DT_FP32);
    }
    {
    scalar_t max_v_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(max_v.addr, max_v_scalar_f32, &max_v.shape, (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
    }
    __ppl_tensor_info kv_s = {.shape = { 1, 32, 1, 512}, .stride = {0}, .addr = 49344, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&kv_s.stride, 0, &kv_s.shape, DT_BFP16);
    __ppl_tensor_info pe_s = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 50368, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_s.stride, 0, &pe_s.shape, DT_BFP16);
    __ppl_tensor_info prob_s = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 20224, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_s.stride, 0, &prob_s.shape, DT_BFP16);
    __ppl_tensor_info score = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score.stride, 0, &score.shape, DT_FP32);
    __ppl_tensor_info score_pe = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 20416, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_pe.stride, 0, &score_pe.shape, DT_FP32);
    __ppl_tensor_info max_prev = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 50496, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&max_prev.stride, 0, &max_prev.shape, DT_FP32);
    __ppl_tensor_info rescale = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&rescale.stride, 0, &rescale.shape, DT_FP32);
    __ppl_tensor_info score_sum = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 20864, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_sum.stride, 0, &score_sum.shape, DT_FP32);
    __ppl_tensor_info prev_scaled = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 20736, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prev_scaled.stride, 0, &prev_scaled.shape, DT_FP32);
    __ppl_tensor_info curr_scaled = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 50560, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&curr_scaled.stride, 0, &curr_scaled.shape, DT_FP32);
    __ppl_tensor_info work0 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 65664, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work0.stride, 0, &work0.shape, DT_FP32);
    __ppl_tensor_info work1 = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work1.stride, 0, &work1.shape, DT_FP32);
    __ppl_tensor_info coeff = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff.stride, 0, &coeff.shape, DT_FP32);
    __ppl_tensor_info table = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table.stride, 0, &table.shape, DT_FP32);
    __ppl_tensor_info max_scaled = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 20800, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&max_scaled.stride, 0, &max_scaled.shape, DT_FP32);
    __ppl_tensor_info work2 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 49216, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work2.stride, 0, &work2.shape, DT_FP32);
    __ppl_tensor_info work3 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work3.stride, 0, &work3.shape, DT_FP32);
    for (int k = 0; k < -1; ++k) {
      tpu_parallel_start(); 
      __ppl_tensor_info kv_gather = {.shape = {1, 32, 1, 512} , .stride = {16384, 512, 512, 1} , .addr = v31.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_s_1 = {.shape = {1, 32, 1, 512} , .stride = kv_s.stride, .addr = kv_s.addr + ((0) * kv_s.stride.c+(0) * kv_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_s.stride.c+(0) * kv_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_s.default_stride};
      tpu_gdma_cpy_S2L(kv_s_1.addr, kv_gather.addr, &kv_s_1.shape, (kv_s_1.default_stride ? NULL : &kv_s_1.stride), (kv_gather.default_stride ? NULL : &kv_gather.stride), DT_BFP16);
      __ppl_tensor_info pe_gather = {.shape = {1, 32, 1, 64} , .stride = {2048, 64, 64, 1} , .addr = v32.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info pe_s_1 = {.shape = {1, 32, 1, 64} , .stride = pe_s.stride, .addr = pe_s.addr + ((0) * pe_s.stride.c+(0) * pe_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_s.stride.c+(0) * pe_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_s.default_stride};
      tpu_gdma_cpy_S2L(pe_s_1.addr, pe_gather.addr, &pe_s_1.shape, (pe_s_1.default_stride ? NULL : &pe_s_1.stride), (pe_gather.default_stride ? NULL : &pe_gather.stride), DT_BFP16);
      {
      scalar_t score_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(score.addr, score_scalar_f32, &score.shape, (score.default_stride ? NULL : &score.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(score.addr, q_abs.addr, kv_s.addr, 1, 512, 32, DT_FP32, DT_BFP16);
      {
      scalar_t score_pe_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(score_pe.addr, score_pe_scalar_f32, &score_pe.shape, (score_pe.default_stride ? NULL : &score_pe.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(score_pe.addr, q_pe.addr, pe_s.addr, 1, 64, 32, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( score.addr, score.addr, score_pe.addr, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), (score_pe.default_stride ? NULL : &score_pe.stride), DT_FP32);
      __ppl_tensor_info max_v_1 = {.shape = {1, 1, 1, 1} , .stride = max_v.stride, .addr = max_v.addr + ((0) * max_v.stride.c+(0) * max_v.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * max_v.stride.c+(0) * max_v.stride.w ) * 4, .unsigned_flag = 0, .default_stride = max_v.default_stride};
      __ppl_tensor_info max_prev_1 = {.shape = {1, 1, 1, 1} , .stride = max_prev.stride, .addr = max_prev.addr + ((0) * max_prev.stride.c+(0) * max_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * max_prev.stride.c+(0) * max_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = max_prev.default_stride};
      tpu_bdc_cpy(max_prev_1.addr, max_v_1.addr, &max_prev_1.shape, (max_prev_1.default_stride ? NULL : &max_prev_1.stride), (max_v_1.default_stride ? NULL : &max_v_1.stride), DT_FP32);
      {
      scalar_t max_v_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
      tpu_bdc_set_C(max_v.addr, max_v_scalar_f32, &max_v.shape, (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
      }
      __ppl_tensor_info tmp_buffer_max = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 19200, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&tmp_buffer_max.stride, 0, &tmp_buffer_max.shape, DT_FP32);
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
        dim4 in_reduce_h = {score.shape.n, score.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {score.shape.n, score.shape.c, 1, eu_num};
        dim4 in_reduce_w = {score.shape.n, score.shape.c, 1, eu_num};
        dim4 out_reduce_w = {score.shape.n, score.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        if (align_w > score.shape.w && align_w == eu_num) {
          dim4 padded_stride = {32, align_w, align_w, 1};
          dim4 padded_shape = {score.shape.n, score.shape.c, 1, align_w};
          dim4 copy_shape = {score.shape.n, score.shape.c, 1, score.shape.w};
          __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = max_v.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
          tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
          dim2 kernel2 = {1, eu_num};
          pad_val.u32 = FP_NEG_MAX(DT_FP32);
          tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
        } else {
          if (align_w > score.shape.w) {
            dim4 fill_shape = {score.shape.n, score.shape.c, 1, align_w - score.shape.w};
            int elem_size = 4;
            int offset = score.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, score.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
            tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
          }
          __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = max_v.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          pad_val.u32 = FP_NEG_MAX(DT_FP32);
          tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
        }
      }
      tpu_bdc_fp_mul_C( prev_scaled.addr, max_prev.addr, (scalar_t){.f32 = 0.0721688}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (max_prev.default_stride ? NULL : &max_prev.stride), DT_FP32);
      tpu_bdc_fp_mul_C( curr_scaled.addr, max_v.addr, (scalar_t){.f32 = 0.0721688}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
      tpu_bdc_fp_sub( rescale.addr, prev_scaled.addr, curr_scaled.addr, &rescale.shape, (rescale.default_stride ? NULL : &rescale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(rescale.addr, rescale.addr, work0.addr, work1.addr, coeff.addr, table.addr, &rescale.shape);
      tpu_bdc_fp_mul_C( score.addr, score.addr, (scalar_t){.f32 = 0.0721688}, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), DT_FP32);
      tpu_bdc_fp_mul_C( max_scaled.addr, max_v.addr, (scalar_t){.f32 = 0.0721688}, &max_scaled.shape, (max_scaled.default_stride ? NULL : &max_scaled.stride), (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
      dim4 max_scaled_stride;
      tpu_aligned_stride(&max_scaled_stride, 0, &max_scaled.shape, DT_FP32);
      max_scaled_stride.w = 0;
      tpu_bdc_fp_sub( score.addr, score.addr, max_scaled.addr, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), &max_scaled_stride, DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(score.addr, score.addr, work2.addr, work3.addr, coeff.addr, table.addr, &score.shape);
      __ppl_tensor_info tmp_buffer_sum = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&tmp_buffer_sum.stride, 0, &tmp_buffer_sum.shape, DT_FP32);
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.f32 = 0};
        dim4 in_reduce_h = {score.shape.n, score.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {score.shape.n, score.shape.c, 1, eu_num};
        dim4 in_reduce_w = {score.shape.n, score.shape.c, 1, eu_num};
        dim4 out_reduce_w = {score.shape.n, score.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        scalar_t scale = {.f32 = (float)1.000000000e+00};
        if (align_w > score.shape.w && align_w == eu_num) {
          dim4 padded_stride = {32, align_w, align_w, 1};
          dim4 padded_shape = {score.shape.n, score.shape.c, 1, align_w};
          dim4 copy_shape = {score.shape.n, score.shape.c, 1, score.shape.w};
          __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = score_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
          tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
          dim2 kernel2 = {1, eu_num};
          tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
        } else {
          if (align_w > score.shape.w) {
            dim4 fill_shape = {score.shape.n, score.shape.c, 1, align_w - score.shape.w};
            int elem_size = 4;
            int offset = score.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, score.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
            tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
          }
          __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = score_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
        }
      }
      __ppl_tensor_info score_1 = {.shape = {1, 1, 1, 32} , .stride = score.stride, .addr = score.addr + ((0) * score.stride.c+(0) * score.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score.stride.c+(0) * score.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score.default_stride};
      __ppl_tensor_info prob_s_1 = {.shape = {1, 1, 1, 32} , .stride = prob_s.stride, .addr = prob_s.addr + ((0) * prob_s.stride.c+(0) * prob_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_s.stride.c+(0) * prob_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_s.default_stride};
      tpu_bdc_cast(prob_s_1.addr, score_1.addr, &prob_s_1.shape, (prob_s_1.default_stride ? NULL : &prob_s_1.stride), (score_1.default_stride ? NULL : &score_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      tpu_bdc_fp_mul( logsum.addr, logsum.addr, rescale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (rescale.default_stride ? NULL : &rescale.stride), DT_FP32);
      tpu_bdc_fp_add( logsum.addr, logsum.addr, score_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (score_sum.default_stride ? NULL : &score_sum.stride), DT_FP32);
      dim4 rescale_stride;
      tpu_aligned_stride(&rescale_stride, 0, &rescale.shape, DT_FP32);
      rescale_stride.w = 0;
      tpu_bdc_fp_mul( acc_o.addr, acc_o.addr, rescale.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &rescale_stride, DT_FP32);
      tpu_bdc_fp_mm(acc_o.addr, prob_s.addr, kv_s.addr, 1, 32, 512, DT_FP32, DT_BFP16, true);
      tpu_parallel_end(); 
    }
    __ppl_tensor_info tmp_buffer_max_1 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 19200, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_1.stride, 0, &tmp_buffer_max_1.shape, DT_FP32);
    __ppl_tensor_info tmp_buffer_sum_1 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_1.stride, 0, &tmp_buffer_sum_1.shape, DT_FP32);
    __ppl_tensor_info kv_gather_1 = {.shape = {1, 32, 1, 512} , .stride = {16384, 512, 512, 1} , .addr = v31.addr + ((0) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_s_2 = {.shape = {1, 32, 1, 512} , .stride = kv_s.stride, .addr = kv_s.addr + ((0) * kv_s.stride.c+(0) * kv_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_s.stride.c+(0) * kv_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_s.default_stride};
    tpu_gdma_cpy_S2L(kv_s_2.addr, kv_gather_1.addr, &kv_s_2.shape, (kv_s_2.default_stride ? NULL : &kv_s_2.stride), (kv_gather_1.default_stride ? NULL : &kv_gather_1.stride), DT_BFP16);
    __ppl_tensor_info pe_gather_1 = {.shape = {1, 32, 1, 64} , .stride = {2048, 64, 64, 1} , .addr = v32.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_s_2 = {.shape = {1, 32, 1, 64} , .stride = pe_s.stride, .addr = pe_s.addr + ((0) * pe_s.stride.c+(0) * pe_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_s.stride.c+(0) * pe_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_s.default_stride};
    tpu_gdma_cpy_S2L(pe_s_2.addr, pe_gather_1.addr, &pe_s_2.shape, (pe_s_2.default_stride ? NULL : &pe_s_2.stride), (pe_gather_1.default_stride ? NULL : &pe_gather_1.stride), DT_BFP16);
    {
    scalar_t score_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score.addr, score_scalar_f32, &score.shape, (score.default_stride ? NULL : &score.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(score.addr, q_abs.addr, kv_s.addr, 1, 512, 32, DT_FP32, DT_BFP16);
    {
    scalar_t score_pe_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_pe.addr, score_pe_scalar_f32, &score_pe.shape, (score_pe.default_stride ? NULL : &score_pe.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(score_pe.addr, q_pe.addr, pe_s.addr, 1, 64, 32, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score.addr, score.addr, score_pe.addr, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), (score_pe.default_stride ? NULL : &score_pe.stride), DT_FP32);
    __ppl_tensor_info max_v_2 = {.shape = {1, 1, 1, 1} , .stride = max_v.stride, .addr = max_v.addr + ((0) * max_v.stride.c+(0) * max_v.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * max_v.stride.c+(0) * max_v.stride.w ) * 4, .unsigned_flag = 0, .default_stride = max_v.default_stride};
    __ppl_tensor_info max_prev_2 = {.shape = {1, 1, 1, 1} , .stride = max_prev.stride, .addr = max_prev.addr + ((0) * max_prev.stride.c+(0) * max_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * max_prev.stride.c+(0) * max_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = max_prev.default_stride};
    tpu_bdc_cpy(max_prev_2.addr, max_v_2.addr, &max_prev_2.shape, (max_prev_2.default_stride ? NULL : &max_prev_2.stride), (max_v_2.default_stride ? NULL : &max_v_2.stride), DT_FP32);
    {
    scalar_t max_v_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(max_v.addr, max_v_scalar_f32, &max_v.shape, (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max_2 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 19200, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max_2.stride, 0, &tmp_buffer_max_2.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 32;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score.shape.n, score.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score.shape.n, score.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score.shape.n, score.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score.shape.n, score.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score.shape.w && align_w == eu_num) {
        dim4 padded_stride = {32, align_w, align_w, 1};
        dim4 padded_shape = {score.shape.n, score.shape.c, 1, align_w};
        dim4 copy_shape = {score.shape.n, score.shape.c, 1, score.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = max_v.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score.shape.w) {
          dim4 fill_shape = {score.shape.n, score.shape.c, 1, align_w - score.shape.w};
          int elem_size = 4;
          int offset = score.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, score.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = max_v.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    tpu_bdc_fp_mul_C( prev_scaled.addr, max_prev.addr, (scalar_t){.f32 = 0.0721688}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (max_prev.default_stride ? NULL : &max_prev.stride), DT_FP32);
    tpu_bdc_fp_mul_C( curr_scaled.addr, max_v.addr, (scalar_t){.f32 = 0.0721688}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
    tpu_bdc_fp_sub( rescale.addr, prev_scaled.addr, curr_scaled.addr, &rescale.shape, (rescale.default_stride ? NULL : &rescale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff.addr);
    tpu_bdc_load_fp32_exp_table(table.addr);
    tpu_bdc_fp32_exp(rescale.addr, rescale.addr, work0.addr, work1.addr, coeff.addr, table.addr, &rescale.shape);
    tpu_bdc_fp_mul_C( score.addr, score.addr, (scalar_t){.f32 = 0.0721688}, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), DT_FP32);
    tpu_bdc_fp_mul_C( max_scaled.addr, max_v.addr, (scalar_t){.f32 = 0.0721688}, &max_scaled.shape, (max_scaled.default_stride ? NULL : &max_scaled.stride), (max_v.default_stride ? NULL : &max_v.stride), DT_FP32);
    dim4 max_scaled_stride_1;
    tpu_aligned_stride(&max_scaled_stride_1, 0, &max_scaled.shape, DT_FP32);
    max_scaled_stride_1.w = 0;
    tpu_bdc_fp_sub( score.addr, score.addr, max_scaled.addr, &score.shape, (score.default_stride ? NULL : &score.stride), (score.default_stride ? NULL : &score.stride), &max_scaled_stride_1, DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff.addr);
    tpu_bdc_load_fp32_exp_table(table.addr);
    tpu_bdc_fp32_exp(score.addr, score.addr, work2.addr, work3.addr, coeff.addr, table.addr, &score.shape);
    __ppl_tensor_info tmp_buffer_sum_2 = {.shape = { 1, 1, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum_2.stride, 0, &tmp_buffer_sum_2.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 32;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score.shape.n, score.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score.shape.n, score.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score.shape.n, score.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score.shape.n, score.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score.shape.w && align_w == eu_num) {
        dim4 padded_stride = {32, align_w, align_w, 1};
        dim4 padded_shape = {score.shape.n, score.shape.c, 1, align_w};
        dim4 copy_shape = {score.shape.n, score.shape.c, 1, score.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = score_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score.shape.w) {
          dim4 fill_shape = {score.shape.n, score.shape.c, 1, align_w - score.shape.w};
          int elem_size = 4;
          int offset = score.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, score.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = score_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum_2.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    __ppl_tensor_info score_2 = {.shape = {1, 1, 1, 32} , .stride = score.stride, .addr = score.addr + ((0) * score.stride.c+(0) * score.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score.stride.c+(0) * score.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score.default_stride};
    __ppl_tensor_info prob_s_2 = {.shape = {1, 1, 1, 32} , .stride = prob_s.stride, .addr = prob_s.addr + ((0) * prob_s.stride.c+(0) * prob_s.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_s.stride.c+(0) * prob_s.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_s.default_stride};
    tpu_bdc_cast(prob_s_2.addr, score_2.addr, &prob_s_2.shape, (prob_s_2.default_stride ? NULL : &prob_s_2.stride), (score_2.default_stride ? NULL : &score_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul( logsum.addr, logsum.addr, rescale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (rescale.default_stride ? NULL : &rescale.stride), DT_FP32);
    tpu_bdc_fp_add( logsum.addr, logsum.addr, score_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (score_sum.default_stride ? NULL : &score_sum.stride), DT_FP32);
    dim4 rescale_stride_1;
    tpu_aligned_stride(&rescale_stride_1, 0, &rescale.shape, DT_FP32);
    rescale_stride_1.w = 0;
    tpu_bdc_fp_mul( acc_o.addr, acc_o.addr, rescale.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &rescale_stride_1, DT_FP32);
    tpu_bdc_fp_mm(acc_o.addr, prob_s.addr, kv_s.addr, 1, 32, 512, DT_FP32, DT_BFP16, true);
    dim4 logsum_stride;
    tpu_aligned_stride(&logsum_stride, 0, &logsum.shape, DT_FP32);
    logsum_stride.w = 0;
    tpu_bdc_fp_div( acc_o.addr, acc_o.addr, logsum.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &logsum_stride, DT_FP32);
    {
    scalar_t value_acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(value_acc.addr, value_acc_scalar_f32, &value_acc.shape, (value_acc.default_stride ? NULL : &value_acc.stride), DT_FP32);
    }
    for (int ib_2 = 0; ib_2 < 4; ++ib_2) {
      __ppl_tensor_info latent_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&latent_shared.stride, 0, &latent_shared.shape, DT_BFP16);
      __ppl_tensor_info weight_fp8_3 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 33024, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_fp8_3.stride, 0, &weight_fp8_3.shape, DT_FP8E4M3);
      __ppl_tensor_info weight_bf16_2 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 16896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_bf16_2.stride, 0, &weight_bf16_2.shape, DT_BFP16);
      __ppl_tensor_info scale_rows_2 = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 33024, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&scale_rows_2.stride, 0, &scale_rows_2.shape, DT_BFP16);
      __ppl_tensor_info weight_dequant_1 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 17408, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_dequant_1.stride, 0, &weight_dequant_1.shape, DT_BFP16);
      __ppl_tensor_info partial = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 33024, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&partial.stride, 0, &partial.shape, DT_FP32);
      __ppl_tensor_info acc_o_1 = {.shape = {1, 1, 1, 128} , .stride = acc_o.stride, .addr = acc_o.addr + ((0) * acc_o.stride.c+((ib_2 * 128)) * acc_o.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * acc_o.stride.c+((ib_2 * 128)) * acc_o.stride.w ) * 4, .unsigned_flag = 0, .default_stride = acc_o.default_stride};
      __ppl_tensor_info latent_shared_1 = {.shape = {1, 1, 1, 128} , .stride = latent_shared.stride, .addr = latent_shared.addr + ((0) * latent_shared.stride.c+(0) * latent_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * latent_shared.stride.c+(0) * latent_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = latent_shared.default_stride};
      tpu_bdc_cast(latent_shared_1.addr, acc_o_1.addr, &latent_shared_1.shape, (latent_shared_1.default_stride ? NULL : &latent_shared_1.stride), (acc_o_1.default_stride ? NULL : &acc_o_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info wukv_1 = {.shape = {1, 128, 1, 128} , .stride = {4194304, 512, 512, 1} , .addr = v21.addr + ((((by * 256) + 128)) * 512+((ib_2 * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((((by * 256) + 128)) * 512+((ib_2 * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_fp8_4 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8_3.stride, .addr = weight_fp8_3.addr + ((0) * weight_fp8_3.stride.c+(0) * weight_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8_3.stride.c+(0) * weight_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8_3.default_stride};
      tpu_gdma_cpy_S2L(weight_fp8_4.addr, wukv_1.addr, &weight_fp8_4.shape, (weight_fp8_4.default_stride ? NULL : &weight_fp8_4.stride), (wukv_1.default_stride ? NULL : &wukv_1.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_fp8_5 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8_3.stride, .addr = weight_fp8_3.addr + ((0) * weight_fp8_3.stride.c+(0) * weight_fp8_3.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8_3.stride.c+(0) * weight_fp8_3.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8_3.default_stride};
      __ppl_tensor_info weight_bf16_3 = {.shape = {1, 128, 1, 128} , .stride = weight_bf16_2.stride, .addr = weight_bf16_2.addr + ((0) * weight_bf16_2.stride.c+(0) * weight_bf16_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_bf16_2.stride.c+(0) * weight_bf16_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_bf16_2.default_stride};
      tpu_bdc_cast(weight_bf16_3.addr, weight_fp8_5.addr, &weight_bf16_3.shape, (weight_bf16_3.default_stride ? NULL : &weight_bf16_3.stride), (weight_fp8_5.default_stride ? NULL : &weight_fp8_5.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info wukv_scale_expanded_1 = {.shape = {1, 128, 1, 1} , .stride = {32768, 4, 4, 1} , .addr = v22.addr + ((((by * 256) + 128)) * 4+(ib_2) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((((by * 256) + 128)) * 4+(ib_2) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info scale_rows_3 = {.shape = {1, 128, 1, 1} , .stride = scale_rows_2.stride, .addr = scale_rows_2.addr + ((0) * scale_rows_2.stride.c+(0) * scale_rows_2.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_rows_2.stride.c+(0) * scale_rows_2.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_rows_2.default_stride};
      tpu_gdma_cpy_S2L(scale_rows_3.addr, wukv_scale_expanded_1.addr, &scale_rows_3.shape, (scale_rows_3.default_stride ? NULL : &scale_rows_3.stride), (wukv_scale_expanded_1.default_stride ? NULL : &wukv_scale_expanded_1.stride), DT_BFP16);
      dim4 scale_rows_2_stride;
      tpu_aligned_stride(&scale_rows_2_stride, 0, &scale_rows_2.shape, DT_BFP16);
      scale_rows_2_stride.w = 0;
      tpu_bdc_fp_mul( weight_dequant_1.addr, weight_bf16_2.addr, scale_rows_2.addr, &weight_dequant_1.shape, (weight_dequant_1.default_stride ? NULL : &weight_dequant_1.stride), (weight_bf16_2.default_stride ? NULL : &weight_bf16_2.stride), &scale_rows_2_stride, DT_BFP16);
      {
      scalar_t partial_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(partial.addr, partial_scalar_f32, &partial.shape, (partial.default_stride ? NULL : &partial.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(partial.addr, latent_shared.addr, weight_dequant_1.addr, 1, 128, 128, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( value_acc.addr, value_acc.addr, partial.addr, &value_acc.shape, (value_acc.default_stride ? NULL : &value_acc.stride), (value_acc.default_stride ? NULL : &value_acc.stride), (partial.default_stride ? NULL : &partial.stride), DT_FP32);
    }
    __ppl_tensor_info value_acc_1 = {.shape = {1, 1, 1, 128} , .stride = value_acc.stride, .addr = value_acc.addr + ((0) * value_acc.stride.c+(0) * value_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * value_acc.stride.c+(0) * value_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = value_acc.default_stride};
    __ppl_tensor_info value_out_1 = {.shape = {1, 1, 1, 128} , .stride = value_out.stride, .addr = value_out.addr + ((0) * value_out.stride.c+(0) * value_out.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * value_out.stride.c+(0) * value_out.stride.w ) * 2, .unsigned_flag = 0, .default_stride = value_out.default_stride};
    tpu_bdc_cast(value_out_1.addr, value_acc_1.addr, &value_out_1.shape, (value_out_1.default_stride ? NULL : &value_out_1.stride), (value_acc_1.default_stride ? NULL : &value_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info value_out_2 = {.shape = {1, 1, 1, 128} , .stride = value_out.stride, .addr = value_out.addr + ((0) * value_out.stride.c+(0) * value_out.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * value_out.stride.c+(0) * value_out.stride.w ) * 2, .unsigned_flag = 0, .default_stride = value_out.default_stride};
    __ppl_tensor_info output = {.shape = {1, 1, 1, 128} , .stride = {4096, 128, 128, 1} , .addr = v34.addr + ((0) * 4096+(by) * 128+(0) * 128+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 4096+(by) * 128+(0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(output.addr, value_out_2.addr, &output.shape, (output.default_stride ? NULL : &output.stride), (value_out_2.default_stride ? NULL : &value_out_2.stride), DT_BFP16);
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
  global_addr_t v16;
  global_addr_t v17;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  main_kernel_inner(api->v1,
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
    api->v16,
    api->v17);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
