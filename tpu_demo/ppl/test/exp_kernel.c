#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1) {
  __ppl_tensor_info v2 = {.shape = {1, 10, 1, 2},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 80,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info scores_scale_shared = {.shape = {1, 5, 1, 1},
                                           .stride = NULL,
                                           .addr = 65536,
                                           .dtype = DT_FP32,
                                           .mode = 2,
                                           .align_mode = 1,
                                           .size = 0,
                                           .unsigned_flag = 0,
                                           .default_stride = true};
  __ppl_tensor_info work0 = {.shape = {1, 5, 1, 1},
                             .stride = NULL,
                             .addr = 49152,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info work1 = {.shape = {1, 5, 1, 1},
                             .stride = NULL,
                             .addr = 32768,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 0,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info coeff = {.shape = {1, 64, 1, 32},
                             .stride = NULL,
                             .addr = 16384,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 128,
                             .unsigned_flag = 0,
                             .default_stride = true};
  __ppl_tensor_info table = {.shape = {1, 64, 1, 192},
                             .stride = NULL,
                             .addr = 0,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .align_mode = 1,
                             .size = 768,
                             .unsigned_flag = 0,
                             .default_stride = true};
  for (int bx = 0; bx < 2; ++bx) {
    for (int by = 0; by < 2; ++by) {
      __ppl_tensor_info scores_scale = {.shape = {1, 5, 1, 1},
                                        .stride = {20, 2, 2, 1},
                                        .addr = v2.addr +
                                                (((by * 5)) * 2 + (bx)*1) * 4,
                                        .dtype = DT_FP32,
                                        .mode = 2,
                                        .size = 1,
                                        .offset = (((by * 5)) * 2 + (bx)*1) * 4,
                                        .unsigned_flag = 0,
                                        .default_stride = false};
      __ppl_tensor_info scores_scale_shared_1 = {.shape = {1, 5, 1, 1},
                                                 .stride = NULL,
                                                 .addr =
                                                     scores_scale_shared.addr,
                                                 .dtype = DT_FP32,
                                                 .mode = 0,
                                                 .size = 1,
                                                 .offset = 0,
                                                 .unsigned_flag = 0,
                                                 .default_stride = true};
      tpu_gdma_cpy_S2L(
          scores_scale_shared_1.addr, scores_scale.addr,
          &scores_scale_shared_1.shape,
          (scores_scale_shared_1.default_stride
               ? NULL
               : &scores_scale_shared_1.stride),
          (scores_scale.default_stride ? NULL : &scores_scale.stride), DT_FP32);
      tpu_bdc_load_fp32_exp_coeff(coeff.addr);
      tpu_bdc_load_fp32_exp_table(table.addr);
      tpu_bdc_fp32_exp(scores_scale_shared.addr, scores_scale_shared.addr,
                       work0.addr, work1.addr, coeff.addr, table.addr,
                       &scores_scale_shared.shape);
      __ppl_tensor_info scores_scale_shared_2 = {.shape = {1, 5, 1, 1},
                                                 .stride = NULL,
                                                 .addr =
                                                     scores_scale_shared.addr,
                                                 .dtype = DT_FP32,
                                                 .mode = 0,
                                                 .size = 1,
                                                 .offset = 0,
                                                 .unsigned_flag = 0,
                                                 .default_stride = true};
      __ppl_tensor_info scores_scale_1 = {
          .shape = {1, 5, 1, 1},
          .stride = {20, 2, 2, 1},
          .addr = v2.addr + (((by * 5)) * 2 + (bx)*1) * 4,
          .dtype = DT_FP32,
          .mode = 2,
          .size = 1,
          .offset = (((by * 5)) * 2 + (bx)*1) * 4,
          .unsigned_flag = 0,
          .default_stride = false};
      tpu_gdma_cpy_L2S(
          scores_scale_1.addr, scores_scale_shared_2.addr,
          &scores_scale_1.shape,
          (scores_scale_1.default_stride ? NULL : &scores_scale_1.stride),
          (scores_scale_shared_2.default_stride
               ? NULL
               : &scores_scale_shared_2.stride),
          DT_FP32);
    }
  }
}

typedef struct {
  global_addr_t v1;
} tpu_kernel_api_main_args_t;
void main_kernel(const void *args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t *)args;
  main(api->v1);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)