#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3,
          global_addr_t v4) {
  __ppl_tensor_info v8 = {.shape = {1, 4, 2, 8},
                          .stride = NULL,
                          .addr = v4,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 128,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1, 4, 2, 8},
                          .stride = NULL,
                          .addr = v2,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 128,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {1, 4, 2, 8},
                          .stride = NULL,
                          .addr = v3,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 128,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1, 4, 2, 8},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 128,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info Q_shared = {.shape = {1, 2, 1, 8},
                                .stride = NULL,
                                .addr = 163840,
                                .dtype = DT_FP16,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info acc_o = {.shape = {1, 2, 1, 8},
                             .stride = NULL,
                             .addr = 98304,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info logsum = {.shape = {1, 2, 1, 1},
                              .stride = NULL,
                              .addr = 81920,
                              .dtype = DT_FP32,
                              .mode = 2,
                              .align_mode = 1,
                              .size = 0,
                              .unsigned_flag = 0,
                              .default_stride = true};
  __ppl_tensor_info scores_max = {.shape = {1, 2, 1, 1},
                                  .stride = NULL,
                                  .addr = 65536,
                                  .dtype = DT_FP32,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 0,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info K_shared = {.shape = {1, 2, 1, 8},
                                .stride = NULL,
                                .addr = 147456,
                                .dtype = DT_FP16,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info acc_s = {.shape = {1, 2, 1, 2},
                             .stride = NULL,
                             .addr = 49152,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info scores_max_prev = {.shape = {1, 2, 1, 1},
                                       .stride = NULL,
                                       .addr = 32768,
                                       .dtype = DT_FP32,
                                       .mode = 2,
                                       .align_mode = 1,
                                       .size = 0,
                                       .unsigned_flag = 0,
                                       .default_stride = true};
  __ppl_tensor_info tmp_buffer_max = {.shape = {1, 2, 1, 32},
                                      .stride = NULL,
                                      .addr = 81920,
                                      .dtype = DT_FP32,
                                      .mode = 2,
                                      .align_mode = 1,
                                      .size = 4,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
  __ppl_tensor_info scores_scale = {.shape = {1, 2, 1, 1},
                                    .stride = NULL,
                                    .addr = 16384,
                                    .dtype = DT_FP32,
                                    .mode = 2,
                                    .align_mode = 1,
                                    .size = 0,
                                    .unsigned_flag = 0,
                                    .default_stride = true};
  __ppl_tensor_info work0 = {.shape = {1, 2, 1, 1},
                             .stride = NULL,
                             .addr = 0,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info work1 = {.shape = {1, 2, 1, 1},
                             .stride = NULL,
                             .addr = 245760,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info coeff = {.shape = {1, 64, 1, 32},
                             .stride = NULL,
                             .addr = 49152,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 128,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info table = {.shape = {1, 64, 1, 192},
                             .stride = NULL,
                             .addr = 16384,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 768,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info work0_1 = {.shape = {1, 2, 1, 2},
                               .stride = NULL,
                               .addr = 229376,
                               .dtype = DT_FP32,
                               .mode = 2,
                               .align_mode = 1,
                               .size = 0,
                               .unsigned_flag = 0,
                               .default_stride = true};
  __ppl_tensor_info work1_1 = {.shape = {1, 2, 1, 2},
                               .stride = NULL,
                               .addr = 212992,
                               .dtype = DT_FP32,
                               .mode = 2,
                               .align_mode = 1,
                               .size = 0,
                               .unsigned_flag = 0,
                               .default_stride = true};
  __ppl_tensor_info coeff_1 = {.shape = {1, 64, 1, 32},
                               .stride = NULL,
                               .addr = 32768,
                               .dtype = DT_FP32,
                               .mode = 2,
                               .align_mode = 1,
                               .size = 128,
                               .unsigned_flag = 0,
                               .default_stride = true};
  __ppl_tensor_info table_1 = {.shape = {1, 64, 1, 192},
                               .stride = NULL,
                               .addr = 0,
                               .dtype = DT_FP32,
                               .mode = 2,
                               .align_mode = 1,
                               .size = 768,
                               .unsigned_flag = 0,
                               .default_stride = true};
  __ppl_tensor_info scores_sum = {.shape = {1, 2, 1, 1},
                                  .stride = NULL,
                                  .addr = 196608,
                                  .dtype = DT_FP32,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 0,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info tmp_buffer_sum = {.shape = {1, 2, 1, 32},
                                      .stride = NULL,
                                      .addr = 65536,
                                      .dtype = DT_FP32,
                                      .mode = 2,
                                      .align_mode = 1,
                                      .size = 4,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
  __ppl_tensor_info acc_s_cast = {.shape = {1, 2, 1, 2},
                                  .stride = NULL,
                                  .addr = 180224,
                                  .dtype = DT_FP16,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 0,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info V_shared = {.shape = {1, 2, 1, 8},
                                .stride = NULL,
                                .addr = 131072,
                                .dtype = DT_FP16,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info O_shared = {.shape = {1, 2, 1, 8},
                                .stride = NULL,
                                .addr = 114688,
                                .dtype = DT_FP16,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  for (int bx = 0; bx < 2; ++bx) {
    for (int by = 0; by < 2; ++by) {
      __ppl_tensor_info Q = {
          .shape = {1, 2, 1, 8},
          .stride = {64, 16, 8, 1},
          .addr = v5.addr + ((0) * 64 + ((bx * 2)) * 16 + (by)*8 + (0) * 1) * 2,
          .dtype = DT_FP16,
          .mode = 2,
          .size = 1,
          .offset = ((0) * 64 + ((bx * 2)) * 16 + (by)*8 + (0) * 1) * 2,
          .unsigned_flag = 0,
          .default_stride = false};
      __ppl_tensor_info Q_shared_1 = {.shape = {1, 2, 1, 8},
                                      .stride = NULL,
                                      .addr = Q_shared.addr,
                                      .dtype = DT_FP16,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      tpu_gdma_cpy_S2L(Q_shared_1.addr, Q.addr, &Q_shared_1.shape,
                       (Q_shared_1.default_stride ? NULL : &Q_shared_1.stride),
                       (Q.default_stride ? NULL : &Q.stride), DT_FP16);
      scalar_t acc_o_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(acc_o.addr, acc_o_scalar_f32, &acc_o.shape,
                    (acc_o.default_stride ? NULL : &acc_o.stride), DT_FP32);
      scalar_t logsum_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(logsum.addr, logsum_scalar_f32, &logsum.shape,
                    (logsum.default_stride ? NULL : &logsum.stride), DT_FP32);
      scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
      tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape,
                    (scores_max.default_stride ? NULL : &scores_max.stride),
                    DT_FP32);
      for (int k = 0; k < 2; ++k) {
        __ppl_tensor_info K = {
            .shape = {1, 2, 1, 8},
            .stride = {64, 16, 8, 1},
            .addr =
                v6.addr + ((0) * 64 + ((k * 2)) * 16 + (by)*8 + (0) * 1) * 2,
            .dtype = DT_FP16,
            .mode = 2,
            .size = 1,
            .offset = ((0) * 64 + ((k * 2)) * 16 + (by)*8 + (0) * 1) * 2,
            .unsigned_flag = 0,
            .default_stride = false};
        __ppl_tensor_info K_shared_1 = {.shape = {1, 2, 1, 8},
                                        .stride = NULL,
                                        .addr = K_shared.addr,
                                        .dtype = DT_FP16,
                                        .mode = 0,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
        tpu_gdma_cpy_S2L(
            K_shared_1.addr, K.addr, &K_shared_1.shape,
            (K_shared_1.default_stride ? NULL : &K_shared_1.stride),
            (K.default_stride ? NULL : &K.stride), DT_FP16);
        scalar_t acc_s_scalar_f32 = {.f32 = 0};
        tpu_bdc_set_C(acc_s.addr, acc_s_scalar_f32, &acc_s.shape,
                      (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
        tpu_bdc_fp_mm_R_trans(acc_s.addr, Q_shared.addr, K_shared.addr, 2, 8, 2,
                              DT_FP32, DT_FP16);
        __ppl_tensor_info scores_max_1 = {.shape = {1, 2, 1, 1},
                                          .stride = NULL,
                                          .addr = scores_max.addr,
                                          .dtype = DT_FP32,
                                          .mode = 0,
                                          .size = 1,
                                          .offset = 0,
                                          .unsigned_flag = 0,
                                          .default_stride = true};
        __ppl_tensor_info scores_max_prev_1 = {.shape = {1, 2, 1, 1},
                                               .stride = NULL,
                                               .addr = scores_max_prev.addr,
                                               .dtype = DT_FP32,
                                               .mode = 0,
                                               .size = 1,
                                               .offset = 0,
                                               .unsigned_flag = 0,
                                               .default_stride = true};
        tpu_bdc_cpy(
            scores_max_prev_1.addr, scores_max_1.addr, &scores_max_prev_1.shape,
            (scores_max_prev_1.default_stride ? NULL
                                              : &scores_max_prev_1.stride),
            (scores_max_1.default_stride ? NULL : &scores_max_1.stride),
            DT_FP32);
        scalar_t scores_max_scalar_f32 = {.f32 = -2147483648};
        tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape,
                      (scores_max.default_stride ? NULL : &scores_max.stride),
                      DT_FP32);
        {
          int eu_num = 32;
          int align_w = 32;
          scalar_t pad_val = {.f16 = FP_NEG_MAX(DT_FP32)};
          if (align_w > acc_s.shape.w) {
            dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1,
                               align_w - acc_s.shape.w};
            int elem_size = 4;
            int offset = acc_s.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape,
                                             .stride = fill_tensor_stride,
                                             .addr = acc_s.addr + offset,
                                             .dtype = DT_FP32,
                                             .mode = 0,
                                             .align_mode = 4,
                                             .size = 1,
                                             .offset = offset,
                                             .unsigned_flag = 0,
                                             .default_stride = false};
            tpu_bdc_set_C(
                fill_tensor.addr, pad_val, &fill_shape,
                (fill_tensor.default_stride ? NULL : &fill_tensor.stride),
                DT_FP32);
          }
          dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num,
                              eu_num};
          dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
          dim2 kernel = {align_w / eu_num, 1};
          padding_t pad = {0, 0, 0, 0};
          dim2 stride = {1, 1};
          dim2 dilation = {1, 1};
          __ppl_tensor_info input_view = {.shape = in_reduce_h,
                                          .stride = {0},
                                          .addr = acc_s.addr,
                                          .dtype = DT_FP32,
                                          .mode = 0,
                                          .align_mode = 1,
                                          .size = 1,
                                          .offset = 0,
                                          .unsigned_flag = 0,
                                          .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h,
                                        .stride = {0},
                                        .addr = tmp_buffer_max.addr,
                                        .dtype = DT_FP32,
                                        .mode = 0,
                                        .align_mode = 1,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
          tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr,
                                &input_view.shape, &kernel, &pad, &stride,
                                &dilation, DT_FP32, pad_val);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w,
                                           .stride = {0},
                                           .addr = scores_max.addr,
                                           .dtype = DT_FP32,
                                           .mode = 0,
                                           .align_mode = 1,
                                           .size = 1,
                                           .offset = 0,
                                           .unsigned_flag = 0,
                                           .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w,
                                         .stride = {0},
                                         .addr = tmp_buffer_max.addr,
                                         .dtype = DT_FP32,
                                         .mode = 0,
                                         .align_mode = 1,
                                         .size = 1,
                                         .offset = 0,
                                         .unsigned_flag = 0,
                                         .default_stride = true};
          pad_val.u32 = FP_NEG_MAX(DT_FP32);
          tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr,
                                &tmp_view2.shape, &kernel2, &pad, &stride,
                                &dilation, DT_FP32, pad_val);
        }
        tpu_bdc_fp_sub(
            scores_scale.addr, scores_max_prev.addr, scores_max.addr,
            &scores_scale.shape,
            (scores_scale.default_stride ? NULL : &scores_scale.stride),
            (scores_max_prev.default_stride ? NULL : &scores_max_prev.stride),
            (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
        tpu_bdc_fp_mul_C(
            scores_scale.addr, scores_scale.addr, (scalar_t){.f32 = 0.51007},
            &scores_scale.shape,
            (scores_scale.default_stride ? NULL : &scores_scale.stride),
            (scores_scale.default_stride ? NULL : &scores_scale.stride),
            DT_FP32);
        tpu_bdc_load_fp32_exp_coeff(coeff.addr);
        tpu_bdc_load_fp32_exp_table(table.addr);
        tpu_bdc_fp32_exp(scores_scale.addr, scores_scale.addr, work0.addr,
                         work1.addr, coeff.addr, table.addr,
                         &scores_scale.shape);
        dim4 scores_max_stride;
        tpu_aligned_stride(&scores_max_stride, 0, &scores_max.shape, DT_FP32);
        scores_max_stride.w = 0;
        tpu_bdc_fp_sub(acc_s.addr, acc_s.addr, scores_max.addr, &acc_s.shape,
                       (acc_s.default_stride ? NULL : &acc_s.stride),
                       (acc_s.default_stride ? NULL : &acc_s.stride),
                       &scores_max_stride, DT_FP32);
        tpu_bdc_fp_mul_C(
            acc_s.addr, acc_s.addr, (scalar_t){.f32 = 0.51007}, &acc_s.shape,
            (acc_s.default_stride ? NULL : &acc_s.stride),
            (acc_s.default_stride ? NULL : &acc_s.stride), DT_FP32);
        tpu_bdc_load_fp32_exp_coeff(coeff_1.addr);
        tpu_bdc_load_fp32_exp_table(table_1.addr);
        tpu_bdc_fp32_exp(acc_s.addr, acc_s.addr, work0_1.addr, work1_1.addr,
                         coeff_1.addr, table_1.addr, &acc_s.shape);
        {
          int eu_num = 32;
          int align_w = 32;
          scalar_t pad_val = {.f32 = 0};
          if (align_w > acc_s.shape.w) {
            dim4 fill_shape = {acc_s.shape.n, acc_s.shape.c, 1,
                               align_w - acc_s.shape.w};
            int elem_size = 4;
            int offset = acc_s.shape.w * elem_size;
            dim4 fill_tensor_stride = {32, align_w, acc_s.shape.w, 1};
            __ppl_tensor_info fill_tensor = {.shape = fill_shape,
                                             .stride = fill_tensor_stride,
                                             .addr = acc_s.addr + offset,
                                             .dtype = DT_FP32,
                                             .mode = 0,
                                             .align_mode = 4,
                                             .size = 1,
                                             .offset = offset,
                                             .unsigned_flag = 0,
                                             .default_stride = false};
            tpu_bdc_set_C(
                fill_tensor.addr, pad_val, &fill_shape,
                (fill_tensor.default_stride ? NULL : &fill_tensor.stride),
                DT_FP32);
          }
          dim4 in_reduce_h = {acc_s.shape.n, acc_s.shape.c, align_w / eu_num,
                              eu_num};
          dim4 out_reduce_h = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 in_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, eu_num};
          dim4 out_reduce_w = {acc_s.shape.n, acc_s.shape.c, 1, 1};
          dim2 kernel = {align_w / eu_num, 1};
          padding_t pad = {0, 0, 0, 0};
          dim2 stride = {1, 1};
          dim2 dilation = {1, 1};
          __ppl_tensor_info input_view = {.shape = in_reduce_h,
                                          .stride = {0},
                                          .addr = acc_s.addr,
                                          .dtype = DT_FP32,
                                          .mode = 0,
                                          .align_mode = 1,
                                          .size = 1,
                                          .offset = 0,
                                          .unsigned_flag = 0,
                                          .default_stride = true};
          __ppl_tensor_info tmp_view = {.shape = out_reduce_h,
                                        .stride = {0},
                                        .addr = tmp_buffer_sum.addr,
                                        .dtype = DT_FP32,
                                        .mode = 0,
                                        .align_mode = 1,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
          scalar_t scale = {.f32 = (float)1.000000000e+00};
          tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr,
                                &input_view.shape, &kernel, &pad, &stride,
                                &dilation, DT_FP32, scale);
          dim2 kernel2 = {1, eu_num};
          __ppl_tensor_info output_view = {.shape = out_reduce_w,
                                           .stride = {0},
                                           .addr = scores_sum.addr,
                                           .dtype = DT_FP32,
                                           .mode = 0,
                                           .align_mode = 1,
                                           .size = 1,
                                           .offset = 0,
                                           .unsigned_flag = 0,
                                           .default_stride = true};
          __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w,
                                         .stride = {0},
                                         .addr = tmp_buffer_sum.addr,
                                         .dtype = DT_FP32,
                                         .mode = 0,
                                         .align_mode = 1,
                                         .size = 1,
                                         .offset = 0,
                                         .unsigned_flag = 0,
                                         .default_stride = true};
          tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr,
                                &tmp_view2.shape, &kernel2, &pad, &stride,
                                &dilation, DT_FP32, scale);
        }
        tpu_bdc_fp_mul(
            logsum.addr, logsum.addr, scores_scale.addr, &logsum.shape,
            (logsum.default_stride ? NULL : &logsum.stride),
            (logsum.default_stride ? NULL : &logsum.stride),
            (scores_scale.default_stride ? NULL : &scores_scale.stride),
            DT_FP32);
        tpu_bdc_fp_add(logsum.addr, logsum.addr, scores_sum.addr, &logsum.shape,
                       (logsum.default_stride ? NULL : &logsum.stride),
                       (logsum.default_stride ? NULL : &logsum.stride),
                       (scores_sum.default_stride ? NULL : &scores_sum.stride),
                       DT_FP32);
        __ppl_tensor_info acc_s_1 = {.shape = {1, 2, 1, 2},
                                     .stride = NULL,
                                     .addr = acc_s.addr,
                                     .dtype = DT_FP32,
                                     .mode = 0,
                                     .size = 1,
                                     .offset = 0,
                                     .unsigned_flag = 0,
                                     .default_stride = true};
        __ppl_tensor_info acc_s_cast_1 = {.shape = {1, 2, 1, 2},
                                          .stride = NULL,
                                          .addr = acc_s_cast.addr,
                                          .dtype = DT_FP16,
                                          .mode = 0,
                                          .size = 1,
                                          .offset = 0,
                                          .unsigned_flag = 0,
                                          .default_stride = true};
        tpu_bdc_cast(
            acc_s_cast_1.addr, acc_s_1.addr, &acc_s_cast_1.shape,
            (acc_s_cast_1.default_stride ? NULL : &acc_s_cast_1.stride),
            (acc_s_1.default_stride ? NULL : &acc_s_1.stride), DT_FP16, DT_FP32,
            RM_HALF_TO_EVEN);
        dim4 scores_scale_stride;
        tpu_aligned_stride(&scores_scale_stride, 0, &scores_scale.shape,
                           DT_FP32);
        scores_scale_stride.w = 0;
        tpu_bdc_fp_mul(acc_o.addr, acc_o.addr, scores_scale.addr, &acc_o.shape,
                       (acc_o.default_stride ? NULL : &acc_o.stride),
                       (acc_o.default_stride ? NULL : &acc_o.stride),
                       &scores_scale_stride, DT_FP32);
        __ppl_tensor_info V = {
            .shape = {1, 2, 1, 8},
            .stride = {64, 16, 8, 1},
            .addr =
                v7.addr + ((0) * 64 + ((k * 2)) * 16 + (by)*8 + (0) * 1) * 2,
            .dtype = DT_FP16,
            .mode = 2,
            .size = 1,
            .offset = ((0) * 64 + ((k * 2)) * 16 + (by)*8 + (0) * 1) * 2,
            .unsigned_flag = 0,
            .default_stride = false};
        __ppl_tensor_info V_shared_1 = {.shape = {1, 2, 1, 8},
                                        .stride = NULL,
                                        .addr = V_shared.addr,
                                        .dtype = DT_FP16,
                                        .mode = 0,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
        tpu_gdma_cpy_S2L(
            V_shared_1.addr, V.addr, &V_shared_1.shape,
            (V_shared_1.default_stride ? NULL : &V_shared_1.stride),
            (V.default_stride ? NULL : &V.stride), DT_FP16);
        tpu_bdc_fp_mm(acc_o.addr, acc_s_cast.addr, V_shared.addr, 2, 2, 8,
                      DT_FP32, DT_FP16, true);
      }
      dim4 logsum_stride;
      tpu_aligned_stride(&logsum_stride, 0, &logsum.shape, DT_FP32);
      logsum_stride.w = 0;
      tpu_bdc_fp_div(acc_o.addr, acc_o.addr, logsum.addr, &acc_o.shape,
                     (acc_o.default_stride ? NULL : &acc_o.stride),
                     (acc_o.default_stride ? NULL : &acc_o.stride),
                     &logsum_stride, DT_FP32);
      __ppl_tensor_info acc_o_1 = {.shape = {1, 2, 1, 8},
                                   .stride = NULL,
                                   .addr = acc_o.addr,
                                   .dtype = DT_FP32,
                                   .mode = 0,
                                   .size = 1,
                                   .offset = 0,
                                   .unsigned_flag = 0,
                                   .default_stride = true};
      __ppl_tensor_info O_shared_1 = {.shape = {1, 2, 1, 8},
                                      .stride = NULL,
                                      .addr = O_shared.addr,
                                      .dtype = DT_FP16,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      tpu_bdc_cast(O_shared_1.addr, acc_o_1.addr, &O_shared_1.shape,
                   (O_shared_1.default_stride ? NULL : &O_shared_1.stride),
                   (acc_o_1.default_stride ? NULL : &acc_o_1.stride), DT_FP16,
                   DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info O_shared_2 = {.shape = {1, 2, 1, 8},
                                      .stride = NULL,
                                      .addr = O_shared.addr,
                                      .dtype = DT_FP16,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      __ppl_tensor_info Output = {
          .shape = {1, 2, 1, 8},
          .stride = {64, 16, 8, 1},
          .addr = v8.addr + ((0) * 64 + ((bx * 2)) * 16 + (by)*8 + (0) * 1) * 2,
          .dtype = DT_FP16,
          .mode = 2,
          .size = 1,
          .offset = ((0) * 64 + ((bx * 2)) * 16 + (by)*8 + (0) * 1) * 2,
          .unsigned_flag = 0,
          .default_stride = false};
      tpu_gdma_cpy_L2S(Output.addr, O_shared_2.addr, &Output.shape,
                       (Output.default_stride ? NULL : &Output.stride),
                       (O_shared_2.default_stride ? NULL : &O_shared_2.stride),
                       DT_FP16);
    }
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
} tpu_kernel_api_main_args_t;
void main_kernel(const void *args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t *)args;
  main(api->v1, api->v2, api->v3, api->v4);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)