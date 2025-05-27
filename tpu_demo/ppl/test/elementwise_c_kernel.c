#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2) {
  __ppl_tensor_info v4 = {.shape = {1, 16, 1, 4},
                          .stride = NULL,
                          .addr = v2,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 256,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v3 = {.shape = {1, 16, 1, 4},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 256,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info A_shared = {.shape = {1, 4, 1, 2},
                                .stride = NULL,
                                .addr = 0,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 0,
                                .unsigned_flag = 0,
                                .default_stride = true};
  for (int bx = 0; bx < 4; ++bx) {
    for (int by = 0; by < 2; ++by) {
      __ppl_tensor_info A = {.shape = {1, 4, 1, 2},
                             .stride = {64, 4, 4, 1},
                             .addr = v3.addr +
                                     (((bx * 4)) * 4 + ((by * 2)) * 1) * 4,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .size = 1,
                             .offset = (((bx * 4)) * 4 + ((by * 2)) * 1) * 4,
                             .unsigned_flag = 0,
                             .default_stride = false};
      __ppl_tensor_info A_shared_1 = {.shape = {1, 4, 1, 2},
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
      tpu_bdc_fp_mul_C(
          A_shared.addr, A_shared.addr, (scalar_t){.f32 = 0.00012207},
          &A_shared.shape, (A_shared.default_stride ? NULL : &A_shared.stride),
          (A_shared.default_stride ? NULL : &A_shared.stride), DT_FP32);
      tpu_bdc_fp_add_C(
          A_shared.addr, A_shared.addr, (scalar_t){.f32 = 1e-12},
          &A_shared.shape, (A_shared.default_stride ? NULL : &A_shared.stride),
          (A_shared.default_stride ? NULL : &A_shared.stride), DT_FP32);
      __ppl_tensor_info A_shared_2 = {.shape = {1, 4, 1, 2},
                                      .stride = NULL,
                                      .addr = A_shared.addr,
                                      .dtype = DT_FP32,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      __ppl_tensor_info B = {.shape = {1, 4, 1, 2},
                             .stride = {64, 4, 4, 1},
                             .addr = v4.addr +
                                     (((bx * 4)) * 4 + ((by * 2)) * 1) * 4,
                             .dtype = DT_FP32,
                             .mode = 2,
                             .size = 1,
                             .offset = (((bx * 4)) * 4 + ((by * 2)) * 1) * 4,
                             .unsigned_flag = 0,
                             .default_stride = false};
      tpu_gdma_cpy_L2S(B.addr, A_shared_2.addr, &B.shape,
                       (B.default_stride ? NULL : &B.stride),
                       (A_shared_2.default_stride ? NULL : &A_shared_2.stride),
                       DT_FP32);
    }
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