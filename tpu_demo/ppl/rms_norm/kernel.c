#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2) {
  __ppl_tensor_info v4 = {.shape = {1, 8192, 1, 8192},
                          .stride = NULL,
                          .addr = v2,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 268435456,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v3 = {.shape = {1, 8192, 1, 8192},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 268435456,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info A_shared = {.shape = {1, 16, 1, 8192},
                                .stride = NULL,
                                .addr = 49152,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 8192,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info A_pow2 = {.shape = {1, 16, 1, 8192},
                              .stride = NULL,
                              .addr = 0,
                              .dtype = DT_FP32,
                              .mode = 2,
                              .align_mode = 1,
                              .size = 8192,
                              .unsigned_flag = 0,
                              .default_stride = true};
  __ppl_tensor_info A_powsum = {.shape = {1, 16, 1, 1},
                                .stride = NULL,
                                .addr = 114688,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info tmp_buffer = {.shape = {1, 16, 1, 32},
                                  .stride = NULL,
                                  .addr = 98304,
                                  .dtype = DT_FP32,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 32,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  for (int bx = 0; bx < 512; ++bx) {
    __ppl_tensor_info A = {.shape = {1, 16, 1, 8192},
                           .stride = {67108864, 8192, 8192, 1},
                           .addr = v3.addr + (((bx * 16)) * 8192 + (0) * 1) * 4,
                           .dtype = DT_FP32,
                           .mode = 2,
                           .size = 1,
                           .offset = (((bx * 16)) * 8192 + (0) * 1) * 4,
                           .unsigned_flag = 0,
                           .default_stride = false};
    __ppl_tensor_info A_shared_1 = {.shape = {1, 16, 1, 8192},
                                    .stride = NULL,
                                    .addr = A_shared.addr,
                                    .dtype = DT_FP32,
                                    .mode = 0,
                                    .size = 1,
                                    .offset = 0,
                                    .unsigned_flag = 0,
                                    .default_stride = true};
    tpu_gdma_cpy_S2L(A_shared_1.addr, A.addr, &A_shared_1.shape,
                     (A_shared_1.default_stride ? NULL : &A_shared_1.stride),
                     (A.default_stride ? NULL : &A.stride), DT_FP32);
    tpu_bdc_fp_mul(A_pow2.addr, A_shared.addr, A_shared.addr, &A_pow2.shape,
                   (A_pow2.default_stride ? NULL : &A_pow2.stride),
                   (A_shared.default_stride ? NULL : &A_shared.stride),
                   (A_shared.default_stride ? NULL : &A_shared.stride),
                   DT_FP32);
    int eu_num = 32;
    int align_w = 8192;
    scalar_t pad_val = {.f32 = 0};
    if (align_w > A_pow2.shape.w) {
      dim4 fill_shape = {A_pow2.shape.n, A_pow2.shape.c, 1,
                         align_w - A_pow2.shape.w};
      int elem_size = 4;
      int offset = A_pow2.shape.w * elem_size;
      dim4 fill_tensor_stride = {8192, align_w, A_pow2.shape.w, 1};
      __ppl_tensor_info fill_tensor = {.shape = fill_shape,
                                       .stride = fill_tensor_stride,
                                       .addr = A_pow2.addr + offset,
                                       .dtype = DT_FP32,
                                       .mode = 0,
                                       .align_mode = 4,
                                       .size = 1,
                                       .offset = offset,
                                       .unsigned_flag = 0,
                                       .default_stride = false};
      tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape,
                    (fill_tensor.default_stride ? NULL : &fill_tensor.stride),
                    DT_FP32);
    }
    dim4 in_reduce_h = {A_pow2.shape.n, A_pow2.shape.c, align_w / eu_num,
                        eu_num};
    dim4 out_reduce_h = {A_pow2.shape.n, A_pow2.shape.c, 1, eu_num};
    dim4 in_reduce_w = {A_pow2.shape.n, A_pow2.shape.c, 1, eu_num};
    dim4 out_reduce_w = {A_pow2.shape.n, A_pow2.shape.c, 1, 1};
    dim2 kernel = {align_w / eu_num, 1};
    padding_t pad = {0, 0, 0, 0};
    dim2 stride = {1, 1};
    dim2 dilation = {1, 1};
    __ppl_tensor_info input_view = {.shape = in_reduce_h,
                                    .stride = {0},
                                    .addr = A_pow2.addr,
                                    .dtype = DT_FP32,
                                    .mode = 0,
                                    .align_mode = 1,
                                    .size = 1,
                                    .offset = 0,
                                    .unsigned_flag = 0,
                                    .default_stride = true};
    __ppl_tensor_info tmp_view = {.shape = out_reduce_h,
                                  .stride = {0},
                                  .addr = tmp_buffer.addr,
                                  .dtype = DT_FP32,
                                  .mode = 0,
                                  .align_mode = 1,
                                  .size = 1,
                                  .offset = 0,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
    scalar_t scale = {.f32 = (float)1.000000000e+00};
    tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape,
                          &kernel, &pad, &stride, &dilation, DT_FP32, scale);
    dim2 kernel2 = {1, eu_num};
    __ppl_tensor_info output_view = {.shape = out_reduce_w,
                                     .stride = {0},
                                     .addr = A_powsum.addr,
                                     .dtype = DT_FP32,
                                     .mode = 0,
                                     .align_mode = 1,
                                     .size = 1,
                                     .offset = 0,
                                     .unsigned_flag = 0,
                                     .default_stride = true};
    __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w,
                                   .stride = {0},
                                   .addr = tmp_buffer.addr,
                                   .dtype = DT_FP32,
                                   .mode = 0,
                                   .align_mode = 1,
                                   .size = 1,
                                   .offset = 0,
                                   .unsigned_flag = 0,
                                   .default_stride = true};
    tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape,
                          &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
    tpu_bdc_fp_mul_C(
        A_powsum.addr, A_powsum.addr, (scalar_t){.f32 = 0.00012207},
        &A_powsum.shape, (A_powsum.default_stride ? NULL : &A_powsum.stride),
        (A_powsum.default_stride ? NULL : &A_powsum.stride), DT_FP32);
    tpu_bdc_fp_add_C(
        A_powsum.addr, A_powsum.addr, (scalar_t){.f32 = 1e-12}, &A_powsum.shape,
        (A_powsum.default_stride ? NULL : &A_powsum.stride),
        (A_powsum.default_stride ? NULL : &A_powsum.stride), DT_FP32);
    tpu_bdc_fp32_rsqrt(A_powsum.addr, A_powsum.addr, &A_powsum.shape);
    dim4 A_powsum_stride;
    tpu_aligned_stride(&A_powsum_stride, 0, &A_powsum.shape, DT_FP32);
    A_powsum_stride.w = 0;
    tpu_bdc_fp_mul(A_shared.addr, A_shared.addr, A_powsum.addr, &A_shared.shape,
                   (A_shared.default_stride ? NULL : &A_shared.stride),
                   (A_shared.default_stride ? NULL : &A_shared.stride),
                   &A_powsum_stride, DT_FP32);
    __ppl_tensor_info A_shared_2 = {.shape = {1, 16, 1, 8192},
                                    .stride = NULL,
                                    .addr = A_shared.addr,
                                    .dtype = DT_FP32,
                                    .mode = 0,
                                    .size = 1,
                                    .offset = 0,
                                    .unsigned_flag = 0,
                                    .default_stride = true};
    __ppl_tensor_info B = {.shape = {1, 16, 1, 8192},
                           .stride = {67108864, 8192, 8192, 1},
                           .addr = v4.addr + (((bx * 16)) * 8192 + (0) * 1) * 4,
                           .dtype = DT_FP32,
                           .mode = 2,
                           .size = 1,
                           .offset = (((bx * 16)) * 8192 + (0) * 1) * 4,
                           .unsigned_flag = 0,
                           .default_stride = false};
    tpu_gdma_cpy_L2S(B.addr, A_shared_2.addr, &B.shape,
                     (B.default_stride ? NULL : &B.stride),
                     (A_shared_2.default_stride ? NULL : &A_shared_2.stride),
                     DT_FP32);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
} tpu_kernel_api_main_args_t;
void main_kernel(const void *args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t *)args;
  main(api->v1, api->v2);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)