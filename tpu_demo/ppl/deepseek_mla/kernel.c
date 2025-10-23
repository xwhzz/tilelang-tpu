#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5) {
  __ppl_tensor_info v10 = {.shape = {4, 1, 8, 32}, .stride = NULL, .addr = v5, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v9 = {.shape = {4, 512, 1, 4}, .stride = NULL, .addr = v4, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v8 = {.shape = {4, 512, 1, 32}, .stride = NULL, .addr = v3, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 131072, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {4, 1, 8, 4}, .stride = NULL, .addr = v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {4, 1, 8, 32}, .stride = NULL, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info Q_shared = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 163840, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info Q_pe_shared = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 229440, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info acc_o = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 98304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info logsum = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 213056, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info scores_max = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 196672, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info K_pe_shared_0 = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 180288, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info K_pe_shared_1 = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 164096, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info KV_shared_0 = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 147456, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info KV_shared_1 = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 131072, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info acc_s = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 147712, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info scores_max_prev = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 131328, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info prev_scaled = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 114944, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info curr_scaled = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 98816, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info scores_scale = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 82432, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work0 = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 66048, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work1 = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 57344, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info coeff = {.shape = { 1, 64, 1, 32}, .stride = NULL, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info table = {.shape = { 1, 64, 1, 192}, .stride = NULL, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info scores_scaled = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 245760, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work0_1 = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 229376, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work1_1 = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 212992, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info scores_sum = {.shape = { 1, 4, 1, 1}, .stride = NULL, .addr = 196608, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info S_shared = {.shape = { 1, 4, 1, 4}, .stride = NULL, .addr = 180224, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info tmp_buffer_max = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info tmp_buffer_sum = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info O_shared = {.shape = { 1, 4, 1, 32}, .stride = NULL, .addr = 114688, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = true};
  for (int bx = 0; bx < 4; ++bx) {
    for (int by = 0; by < 2; ++by) {
      __ppl_tensor_info Q = {.shape = {1, 1, 4, 32} , .stride = {256, 256, 32, 1} , .addr = v6.addr + ((bx) * 256+(0) * 256+((by * 4)) * 32+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 256+(0) * 256+((by * 4)) * 32+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info Q_shared_1 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = Q_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(Q_shared_1.addr, Q.addr, &Q_shared_1.shape, (Q_shared_1.default_stride ? NULL : &Q_shared_1.stride), (Q.default_stride ? NULL : &Q.stride), DT_FP16);
      __ppl_tensor_info Q_pe = {.shape = {1, 1, 4, 4} , .stride = {32, 32, 4, 1} , .addr = v7.addr + ((bx) * 32+(0) * 32+((by * 4)) * 4+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 32+(0) * 32+((by * 4)) * 4+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info Q_pe_shared_1 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = Q_pe_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(Q_pe_shared_1.addr, Q_pe.addr, &Q_pe_shared_1.shape, (Q_pe_shared_1.default_stride ? NULL : &Q_pe_shared_1.stride), (Q_pe.default_stride ? NULL : &Q_pe.stride), DT_FP16);
      {
      scalar_t acc_o_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(acc_o.addr, acc_o_scalar_f32, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), DT_FP32);
      }
      {
      scalar_t logsum_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(logsum.addr, logsum_scalar_f32, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), DT_FP32);
      }
      {
      scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
      tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      }
      __ppl_tensor_info K_pe_shared = K_pe_shared_0;
      __ppl_tensor_info K_pe = {.shape = {1, 4, 1, 4} , .stride = {2048, 4, 4, 1} , .addr = v9.addr + ((bx) * 2048+(0) * 4+(0) * 4+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 2048+(0) * 4+(0) * 4+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info K_pe_shared_2 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = K_pe_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(K_pe_shared_2.addr, K_pe.addr, &K_pe_shared_2.shape, (K_pe_shared_2.default_stride ? NULL : &K_pe_shared_2.stride), (K_pe.default_stride ? NULL : &K_pe.stride), DT_FP16);
      __ppl_tensor_info KV_shared = KV_shared_0;
      __ppl_tensor_info KV = {.shape = {1, 4, 1, 32} , .stride = {16384, 32, 32, 1} , .addr = v8.addr + ((bx) * 16384+(0) * 32+(0) * 32+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 16384+(0) * 32+(0) * 32+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info KV_shared_2 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = KV_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(KV_shared_2.addr, KV.addr, &KV_shared_2.shape, (KV_shared_2.default_stride ? NULL : &KV_shared_2.stride), (KV.default_stride ? NULL : &KV.stride), DT_FP16);
      __ppl_tensor_info K_pe_shared_3 = K_pe_shared_1;
      __ppl_tensor_info K_pe_1 = {.shape = {1, 4, 1, 4} , .stride = {2048, 4, 4, 1} , .addr = v9.addr + ((bx) * 2048+(4) * 4+(0) * 4+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 2048+(4) * 4+(0) * 4+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info K_pe_shared_4 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = K_pe_shared_3.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(K_pe_shared_4.addr, K_pe_1.addr, &K_pe_shared_4.shape, (K_pe_shared_4.default_stride ? NULL : &K_pe_shared_4.stride), (K_pe_1.default_stride ? NULL : &K_pe_1.stride), DT_FP16);
      __ppl_tensor_info KV_shared_3 = KV_shared_1;
      __ppl_tensor_info KV_1 = {.shape = {1, 4, 1, 32} , .stride = {16384, 32, 32, 1} , .addr = v8.addr + ((bx) * 16384+(4) * 32+(0) * 32+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 16384+(4) * 32+(0) * 32+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info KV_shared_4 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = KV_shared_3.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_gdma_cpy_S2L(KV_shared_4.addr, KV_1.addr, &KV_shared_4.shape, (KV_shared_4.default_stride ? NULL : &KV_shared_4.stride), (KV_1.default_stride ? NULL : &KV_1.stride), DT_FP16);
      for (int k = 0; k < 126; ++k) {
        tpu_parallel_start(); 
        {
        scalar_t acc_s_scalar_f32 = {.f32 = 0};
        tpu_bdc_set_C(acc_s.addr, acc_s_scalar_f32, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
        }
        __ppl_tensor_info  condval;
        if (((k % 2) == 0)) {
          condval = KV_shared_0;
        } else {
          condval = KV_shared_1;
        }
        __ppl_tensor_info KV_shared_5 = condval;
        tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_shared.addr, KV_shared_5.addr, 4, 32, 4, DT_FP32, DT_FP16);
        __ppl_tensor_info  condval_1;
        if (((k % 2) == 0)) {
          condval_1 = K_pe_shared_0;
        } else {
          condval_1 = K_pe_shared_1;
        }
        __ppl_tensor_info K_pe_shared_5 = condval_1;
        tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_pe_shared.addr, K_pe_shared_5.addr, 4, 4, 4, DT_FP32, DT_FP16);
        __ppl_tensor_info  condval_2;
        if (((k % 2) == 0)) {
          condval_2 = K_pe_shared_0;
        } else {
          condval_2 = K_pe_shared_1;
        }
        __ppl_tensor_info K_pe_shared_6 = condval_2;
        __ppl_tensor_info K_pe_2 = {.shape = {1, 4, 1, 4} , .stride = {2048, 4, 4, 1} , .addr = v9.addr + ((bx) * 2048+(((k * 4) + 8)) * 4+(0) * 4+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 2048+(((k * 4) + 8)) * 4+(0) * 4+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info K_pe_shared_7 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = K_pe_shared_6.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_gdma_cpy_S2L(K_pe_shared_7.addr, K_pe_2.addr, &K_pe_shared_7.shape, (K_pe_shared_7.default_stride ? NULL : &K_pe_shared_7.stride), (K_pe_2.default_stride ? NULL : &K_pe_2.stride), DT_FP16);
        __ppl_tensor_info scores_max_1 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info scores_max_prev_1 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max_prev.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_cpy(scores_max_prev_1.addr, scores_max_1.addr, &scores_max_prev_1.shape, (scores_max_prev_1.default_stride ? NULL : &scores_max_prev_1.stride), (scores_max_1.default_stride ? NULL : &scores_max_1.stride), DT_FP32);
        {
        scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
        tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
        }
        {
          int eu_num = 32;
          int align_w = 32;
          scalar_t pad_val = {.f16 = FP_NEG_MAX(DT_FP32)};
          if (align_w > acc_s.shape.w) {
            dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
            int elem_size = 4;
            int offset = acc_s.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
            tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
          }
          dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
          dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
          dim2 kernel = {align_w / eu_num, 1};
          padding_t pad = {0, 0, 0, 0};
          dim2 stride = {1, 1};
          dim2 dilation = {1, 1};
          __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          pad_val.u32 = FP_NEG_MAX(DT_FP32);
          tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
        }
        tpu_bdc_fp_mul_C( prev_scaled.addr, scores_max_prev.addr, (scalar_t){.f32 = 0.240449}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (scores_max_prev.default_stride ? NULL : &scores_max_prev.stride), DT_FP32);
        tpu_bdc_fp_mul_C( curr_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
        tpu_bdc_fp_sub( scores_scale.addr, prev_scaled.addr, curr_scaled.addr, &scores_scale.shape, (scores_scale.default_stride ? NULL : &scores_scale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
        tpu_bdc_load_fp32_exp_coeff(coeff.addr);
        tpu_bdc_load_fp32_exp_table(table.addr);
        tpu_bdc_fp32_exp(scores_scale.addr, scores_scale.addr, work0.addr, work1.addr, coeff.addr, table.addr, &scores_scale.shape);
        tpu_bdc_fp_mul_C( acc_s.addr, acc_s.addr, (scalar_t){.f32 = 0.240449}, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
        tpu_bdc_fp_mul_C( scores_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &scores_scaled.shape, (scores_scaled.default_stride ? NULL : &scores_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
        dim4 scores_scaled_stride;
        tpu_aligned_stride(&scores_scaled_stride, 0, &scores_scaled.shape, DT_FP32);
        scores_scaled_stride.w = 0;
        tpu_bdc_fp_sub( acc_s.addr, acc_s.addr, scores_scaled.addr, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), &scores_scaled_stride, DT_FP32);
        tpu_bdc_load_fp32_exp_coeff(coeff.addr);
        tpu_bdc_load_fp32_exp_table(table.addr);
        tpu_bdc_fp32_exp(acc_s.addr, acc_s.addr, work0_1.addr, work1_1.addr, coeff.addr, table.addr, &acc_s.shape);
        {
          int eu_num = 32;
          int align_w = 32;
          scalar_t pad_val = {.f32 = 0};
          if (align_w > acc_s.shape.w) {
            dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
            int elem_size = 4;
            int offset = acc_s.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
            tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
          }
          dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
          dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
          dim2 kernel = {align_w / eu_num, 1};
          padding_t pad = {0, 0, 0, 0};
          dim2 stride = {1, 1};
          dim2 dilation = {1, 1};
          __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          scalar_t scale = {.f32 = (float)1.000000000e+00};
          tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
          tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
        }
        __ppl_tensor_info acc_s_1 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info S_shared_1 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = S_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_cast(S_shared_1.addr, acc_s_1.addr, &S_shared_1.shape, (S_shared_1.default_stride ? NULL : &S_shared_1.stride), (acc_s_1.default_stride ? NULL : &acc_s_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
        tpu_bdc_fp_mul( logsum.addr, logsum.addr, scores_scale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_scale.default_stride ? NULL : &scores_scale.stride), DT_FP32);
        tpu_bdc_fp_add( logsum.addr, logsum.addr, scores_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_sum.default_stride ? NULL : &scores_sum.stride), DT_FP32);
        dim4 scores_scale_stride;
        tpu_aligned_stride(&scores_scale_stride, 0, &scores_scale.shape, DT_FP32);
        scores_scale_stride.w = 0;
        tpu_bdc_fp_mul( acc_o.addr, acc_o.addr, scores_scale.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &scores_scale_stride, DT_FP32);
        __ppl_tensor_info  condval_3;
        if (((k % 2) == 0)) {
          condval_3 = KV_shared_0;
        } else {
          condval_3 = KV_shared_1;
        }
        __ppl_tensor_info KV_shared_6 = condval_3;
        tpu_bdc_fp_mm(acc_o.addr, S_shared.addr, KV_shared_6.addr, 4, 4, 32, DT_FP32, DT_FP16, true);
        __ppl_tensor_info  condval_4;
        if (((k % 2) == 0)) {
          condval_4 = KV_shared_0;
        } else {
          condval_4 = KV_shared_1;
        }
        __ppl_tensor_info KV_shared_7 = condval_4;
        __ppl_tensor_info KV_2 = {.shape = {1, 4, 1, 32} , .stride = {16384, 32, 32, 1} , .addr = v8.addr + ((bx) * 16384+(((k * 4) + 8)) * 32+(0) * 32+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 16384+(((k * 4) + 8)) * 32+(0) * 32+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info KV_shared_8 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = KV_shared_7.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_gdma_cpy_S2L(KV_shared_8.addr, KV_2.addr, &KV_shared_8.shape, (KV_shared_8.default_stride ? NULL : &KV_shared_8.stride), (KV_2.default_stride ? NULL : &KV_2.stride), DT_FP16);
        tpu_parallel_end(); 
      }
      {
      scalar_t acc_s_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(acc_s.addr, acc_s_scalar_f32, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
      }
      __ppl_tensor_info KV_shared_9 = KV_shared_0;
      tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_shared.addr, KV_shared_9.addr, 4, 32, 4, DT_FP32, DT_FP16);
      __ppl_tensor_info K_pe_shared_8 = K_pe_shared_0;
      tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_pe_shared.addr, K_pe_shared_8.addr, 4, 4, 4, DT_FP32, DT_FP16);
      __ppl_tensor_info scores_max_2 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info scores_max_prev_2 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max_prev.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_cpy(scores_max_prev_2.addr, scores_max_2.addr, &scores_max_prev_2.shape, (scores_max_prev_2.default_stride ? NULL : &scores_max_prev_2.stride), (scores_max_2.default_stride ? NULL : &scores_max_2.stride), DT_FP32);
      {
      scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
      tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      }
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.f16 = FP_NEG_MAX(DT_FP32)};
        if (align_w > acc_s.shape.w) {
          dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
          int elem_size = 4;
          int offset = acc_s.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
      tpu_bdc_fp_mul_C( prev_scaled.addr, scores_max_prev.addr, (scalar_t){.f32 = 0.240449}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (scores_max_prev.default_stride ? NULL : &scores_max_prev.stride), DT_FP32);
      tpu_bdc_fp_mul_C( curr_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      tpu_bdc_fp_sub( scores_scale.addr, prev_scaled.addr, curr_scaled.addr, &scores_scale.shape, (scores_scale.default_stride ? NULL : &scores_scale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(scores_scale.addr, scores_scale.addr, work0.addr, work1.addr, coeff.addr, table.addr, &scores_scale.shape);
      tpu_bdc_fp_mul_C( acc_s.addr, acc_s.addr, (scalar_t){.f32 = 0.240449}, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
      tpu_bdc_fp_mul_C( scores_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &scores_scaled.shape, (scores_scaled.default_stride ? NULL : &scores_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      dim4 scores_scaled_stride;
      tpu_aligned_stride(&scores_scaled_stride, 0, &scores_scaled.shape, DT_FP32);
      scores_scaled_stride.w = 0;
      tpu_bdc_fp_sub( acc_s.addr, acc_s.addr, scores_scaled.addr, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), &scores_scaled_stride, DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(acc_s.addr, acc_s.addr, work0_1.addr, work1_1.addr, coeff.addr, table.addr, &acc_s.shape);
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.f32 = 0};
        if (align_w > acc_s.shape.w) {
          dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
          int elem_size = 4;
          int offset = acc_s.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        scalar_t scale = {.f32 = (float)1.000000000e+00};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
      __ppl_tensor_info acc_s_2 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info S_shared_2 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = S_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_cast(S_shared_2.addr, acc_s_2.addr, &S_shared_2.shape, (S_shared_2.default_stride ? NULL : &S_shared_2.stride), (acc_s_2.default_stride ? NULL : &acc_s_2.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      tpu_bdc_fp_mul( logsum.addr, logsum.addr, scores_scale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_scale.default_stride ? NULL : &scores_scale.stride), DT_FP32);
      tpu_bdc_fp_add( logsum.addr, logsum.addr, scores_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_sum.default_stride ? NULL : &scores_sum.stride), DT_FP32);
      dim4 scores_scale_stride;
      tpu_aligned_stride(&scores_scale_stride, 0, &scores_scale.shape, DT_FP32);
      scores_scale_stride.w = 0;
      tpu_bdc_fp_mul( acc_o.addr, acc_o.addr, scores_scale.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &scores_scale_stride, DT_FP32);
      __ppl_tensor_info KV_shared_10 = KV_shared_0;
      tpu_bdc_fp_mm(acc_o.addr, S_shared.addr, KV_shared_10.addr, 4, 4, 32, DT_FP32, DT_FP16, true);
      {
      scalar_t acc_s_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(acc_s.addr, acc_s_scalar_f32, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
      }
      __ppl_tensor_info KV_shared_11 = KV_shared_1;
      tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_shared.addr, KV_shared_11.addr, 4, 32, 4, DT_FP32, DT_FP16);
      __ppl_tensor_info K_pe_shared_9 = K_pe_shared_1;
      tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_pe_shared.addr, K_pe_shared_9.addr, 4, 4, 4, DT_FP32, DT_FP16);
      __ppl_tensor_info scores_max_3 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info scores_max_prev_3 = {.shape = {1, 4, 1, 1}, .stride = NULL, .addr = scores_max_prev.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_cpy(scores_max_prev_3.addr, scores_max_3.addr, &scores_max_prev_3.shape, (scores_max_prev_3.default_stride ? NULL : &scores_max_prev_3.stride), (scores_max_3.default_stride ? NULL : &scores_max_3.stride), DT_FP32);
      {
      scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
      tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      }
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.f16 = FP_NEG_MAX(DT_FP32)};
        if (align_w > acc_s.shape.w) {
          dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
          int elem_size = 4;
          int offset = acc_s.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
      tpu_bdc_fp_mul_C( prev_scaled.addr, scores_max_prev.addr, (scalar_t){.f32 = 0.240449}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (scores_max_prev.default_stride ? NULL : &scores_max_prev.stride), DT_FP32);
      tpu_bdc_fp_mul_C( curr_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      tpu_bdc_fp_sub( scores_scale.addr, prev_scaled.addr, curr_scaled.addr, &scores_scale.shape, (scores_scale.default_stride ? NULL : &scores_scale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(scores_scale.addr, scores_scale.addr, work0.addr, work1.addr, coeff.addr, table.addr, &scores_scale.shape);
      tpu_bdc_fp_mul_C( acc_s.addr, acc_s.addr, (scalar_t){.f32 = 0.240449}, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
      tpu_bdc_fp_mul_C( scores_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.240449}, &scores_scaled.shape, (scores_scaled.default_stride ? NULL : &scores_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
      dim4 scores_scaled_stride;
      tpu_aligned_stride(&scores_scaled_stride, 0, &scores_scaled.shape, DT_FP32);
      scores_scaled_stride.w = 0;
      tpu_bdc_fp_sub( acc_s.addr, acc_s.addr, scores_scaled.addr, &acc_s.shape, (acc_s.default_stride ? NULL : &acc_s.stride), (acc_s.default_stride ? NULL : &acc_s.stride), &scores_scaled_stride, DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(acc_s.addr, acc_s.addr, work0_1.addr, work1_1.addr, coeff.addr, table.addr, &acc_s.shape);
      {
        int eu_num = 32;
        int align_w = 32;
        scalar_t pad_val = {.f32 = 0};
        if (align_w > acc_s.shape.w) {
          dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1, align_w - acc_s.shape.w};
          int elem_size = 4;
          int offset = acc_s.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = acc_s.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num, eu_num};
        dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
        dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
        dim2 kernel = {align_w / eu_num, 1};
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        scalar_t scale = {.f32 = (float)1.000000000e+00};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
      __ppl_tensor_info acc_s_3 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = acc_s.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info S_shared_3 = {.shape = {1, 4, 1, 4}, .stride = NULL, .addr = S_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_cast(S_shared_3.addr, acc_s_3.addr, &S_shared_3.shape, (S_shared_3.default_stride ? NULL : &S_shared_3.stride), (acc_s_3.default_stride ? NULL : &acc_s_3.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      tpu_bdc_fp_mul( logsum.addr, logsum.addr, scores_scale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_scale.default_stride ? NULL : &scores_scale.stride), DT_FP32);
      tpu_bdc_fp_add( logsum.addr, logsum.addr, scores_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_sum.default_stride ? NULL : &scores_sum.stride), DT_FP32);
      dim4 scores_scale_stride;
      tpu_aligned_stride(&scores_scale_stride, 0, &scores_scale.shape, DT_FP32);
      scores_scale_stride.w = 0;
      tpu_bdc_fp_mul( acc_o.addr, acc_o.addr, scores_scale.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &scores_scale_stride, DT_FP32);
      __ppl_tensor_info KV_shared_12 = KV_shared_1;
      tpu_bdc_fp_mm(acc_o.addr, S_shared.addr, KV_shared_12.addr, 4, 4, 32, DT_FP32, DT_FP16, true);
      dim4 logsum_stride;
      tpu_aligned_stride(&logsum_stride, 0, &logsum.shape, DT_FP32);
      logsum_stride.w = 0;
      tpu_bdc_fp_div( acc_o.addr, acc_o.addr, logsum.addr, &acc_o.shape, (acc_o.default_stride ? NULL : &acc_o.stride), (acc_o.default_stride ? NULL : &acc_o.stride), &logsum_stride, DT_FP32);
      __ppl_tensor_info acc_o_1 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = acc_o.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info O_shared_1 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = O_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_cast(O_shared_1.addr, acc_o_1.addr, &O_shared_1.shape, (O_shared_1.default_stride ? NULL : &O_shared_1.stride), (acc_o_1.default_stride ? NULL : &acc_o_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info O_shared_2 = {.shape = {1, 4, 1, 32}, .stride = NULL, .addr = O_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info Output = {.shape = {1, 1, 4, 32} , .stride = {256, 256, 32, 1} , .addr = v10.addr + ((bx) * 256+(0) * 256+((by * 4)) * 32+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((bx) * 256+(0) * 256+((by * 4)) * 32+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(Output.addr, O_shared_2.addr, &Output.shape, (Output.default_stride ? NULL : &Output.stride), (O_shared_2.default_stride ? NULL : &O_shared_2.stride), DT_FP16);
    }
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
  global_addr_t v5;
} tpu_kernel_api_main_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
  main(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)