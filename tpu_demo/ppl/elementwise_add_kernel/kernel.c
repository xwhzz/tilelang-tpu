#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3) {
  __ppl_tensor_info v6 = {.shape = {1, 512, 1, 1024},
                          .stride = NULL,
                          .addr = v3,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 2097152,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1, 512, 1, 1024},
                          .stride = NULL,
                          .addr = v2,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 2097152,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1, 512, 1, 1024},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 2097152,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info A_shared = {.shape = {1, 128, 1, 256},
                                .stride = NULL,
                                .addr = 32768,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 2048,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info B_shared = {.shape = {1, 128, 1, 256},
                                .stride = NULL,
                                .addr = 16384,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 2048,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info C_shared = {.shape = {1, 128, 1, 256},
                                .stride = NULL,
                                .addr = 0,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 2048,
                                .unsigned_flag = 0,
                                .default_stride = true};
  for (int bx = 0; bx < 4; ++bx) {
    for (int by = 0; by < 4; ++by) {
      __ppl_tensor_info A = {
          .shape = {1, 128, 1, 256},
          .stride = {524288, 1024, 1024, 1},
          .addr = v4.addr + (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .dtype = DT_FP32,
          .mode = 2,
          .size = 1,
          .offset = (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .unsigned_flag = 0,
          .default_stride = false};
      __ppl_tensor_info A_shared_1 = {.shape = {1, 128, 1, 256},
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
      __ppl_tensor_info B = {
          .shape = {1, 128, 1, 256},
          .stride = {524288, 1024, 1024, 1},
          .addr = v5.addr + (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .dtype = DT_FP32,
          .mode = 2,
          .size = 1,
          .offset = (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .unsigned_flag = 0,
          .default_stride = false};
      __ppl_tensor_info B_shared_1 = {.shape = {1, 128, 1, 256},
                                      .stride = NULL,
                                      .addr = B_shared.addr,
                                      .dtype = DT_FP32,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      tpu_gdma_cpy_S2L(B_shared_1.addr, B.addr, &B_shared_1.shape,
                       (B_shared_1.default_stride ? NULL : &B_shared_1.stride),
                       (B.default_stride ? NULL : &B.stride), DT_FP32);
      tpu_bdc_fp_add(
          C_shared.addr, A_shared.addr, B_shared.addr, &C_shared.shape,
          (C_shared.default_stride ? NULL : &C_shared.stride),
          (A_shared.default_stride ? NULL : &A_shared.stride),
          (B_shared.default_stride ? NULL : &B_shared.stride), DT_FP32);
      __ppl_tensor_info C_shared_1 = {.shape = {1, 128, 1, 256},
                                      .stride = NULL,
                                      .addr = C_shared.addr,
                                      .dtype = DT_FP32,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      __ppl_tensor_info C = {
          .shape = {1, 128, 1, 256},
          .stride = {524288, 1024, 1024, 1},
          .addr = v6.addr + (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .dtype = DT_FP32,
          .mode = 2,
          .size = 1,
          .offset = (((by * 128)) * 1024 + ((bx * 256)) * 1) * 4,
          .unsigned_flag = 0,
          .default_stride = false};
      tpu_gdma_cpy_L2S(C.addr, C_shared_1.addr, &C.shape,
                       (C.default_stride ? NULL : &C.stride),
                       (C_shared_1.default_stride ? NULL : &C_shared_1.stride),
                       DT_FP32);
    }
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
} tpu_kernel_api_main_args_t;
void main_kernel(const void *args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t *)args;
  main(api->v1, api->v2, api->v3);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)