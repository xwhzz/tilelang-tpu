#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
                           DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
                           DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
                           DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3) {
  __ppl_tensor_info v5 = {.shape = {1, 786, 1, 786},
                          .stride = NULL,
                          .addr = v2,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 1235592,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1, 384, 1, 786},
                          .stride = NULL,
                          .addr = v3,
                          .dtype = DT_FP32,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 1207296,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1, 384, 1, 786},
                          .stride = NULL,
                          .addr = v1,
                          .dtype = DT_FP16,
                          .mode = 2,
                          .align_mode = 0,
                          .size = 603648,
                          .unsigned_flag = 0,
                          .default_stride = true};
  __ppl_tensor_info C_shared = {.shape = {1, 128, 1, 128},
                                .stride = NULL,
                                .addr = 0,
                                .dtype = DT_FP32,
                                .mode = 2,
                                .align_mode = 1,
                                .size = 1024,
                                .unsigned_flag = 0,
                                .default_stride = true};
  __ppl_tensor_info A_shared_0 = {.shape = {1, 128, 1, 128},
                                  .stride = NULL,
                                  .addr = 65536,
                                  .dtype = DT_FP16,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 512,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info B_shared_0 = {.shape = {1, 128, 1, 128},
                                  .stride = NULL,
                                  .addr = 49152,
                                  .dtype = DT_FP16,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 512,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info A_shared_1 = {.shape = {1, 128, 1, 128},
                                  .stride = NULL,
                                  .addr = 32768,
                                  .dtype = DT_FP16,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 512,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  __ppl_tensor_info B_shared_1 = {.shape = {1, 128, 1, 128},
                                  .stride = NULL,
                                  .addr = 16384,
                                  .dtype = DT_FP16,
                                  .mode = 2,
                                  .align_mode = 1,
                                  .size = 512,
                                  .unsigned_flag = 0,
                                  .default_stride = true};
  for (int bx = 0; bx < 7; ++bx) {
    for (int by = 0; by < 3; ++by) {
      scalar_t C_shared_scalar_f32 = {.f32 = 0};
      tpu_bdc_set_C(C_shared.addr, C_shared_scalar_f32, &C_shared.shape,
                    (C_shared.default_stride ? NULL : &C_shared.stride),
                    DT_FP32);
      __ppl_tensor_info A_shared = A_shared_0;
      __ppl_tensor_info A = {.shape = {1, 128, 1, 128},
                             .stride = {301824, 786, 786, 1},
                             .addr =
                                 v4.addr + (((by * 128)) * 786 + (0) * 1) * 2,
                             .dtype = DT_FP16,
                             .mode = 2,
                             .size = 1,
                             .offset = (((by * 128)) * 786 + (0) * 1) * 2,
                             .unsigned_flag = 0,
                             .default_stride = false};
      __ppl_tensor_info A_shared_2 = {.shape = {1, 128, 1, 128},
                                      .stride = NULL,
                                      .addr = A_shared.addr,
                                      .dtype = DT_FP16,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      tpu_gdma_cpy_S2L(A_shared_2.addr, A.addr, &A_shared_2.shape,
                       (A_shared_2.default_stride ? NULL : &A_shared_2.stride),
                       (A.default_stride ? NULL : &A.stride), DT_FP16);
      __ppl_tensor_info B_shared = B_shared_0;
      __ppl_tensor_info B = {.shape = {1, 128, 1, 128},
                             .stride = {617796, 786, 786, 1},
                             .addr =
                                 v5.addr + ((0) * 786 + ((bx * 128)) * 1) * 2,
                             .dtype = DT_FP16,
                             .mode = 2,
                             .size = 1,
                             .offset = ((0) * 786 + ((bx * 128)) * 1) * 2,
                             .unsigned_flag = 0,
                             .default_stride = false};
      __ppl_tensor_info B_shared_2 = {.shape = {1, 128, 1, 128},
                                      .stride = NULL,
                                      .addr = B_shared.addr,
                                      .dtype = DT_FP16,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      tpu_gdma_cpy_S2L(B_shared_2.addr, B.addr, &B_shared_2.shape,
                       (B_shared_2.default_stride ? NULL : &B_shared_2.stride),
                       (B.default_stride ? NULL : &B.stride), DT_FP16);
      for (int k = 0; k < 6; ++k) {
        tpu_parallel_start();
        __ppl_tensor_info condval;
        if ((((k + 1) % 2) == 0)) {
          condval = A_shared_0;
        } else {
          condval = A_shared_1;
        }
        __ppl_tensor_info A_shared_3 = condval;
        __ppl_tensor_info A_1 = {
            .shape = {1, 128, 1, 128},
            .stride = {301824, 786, 786, 1},
            .addr =
                v4.addr + (((by * 128)) * 786 + (((k * 128) + 128)) * 1) * 2,
            .dtype = DT_FP16,
            .mode = 2,
            .size = 1,
            .offset = (((by * 128)) * 786 + (((k * 128) + 128)) * 1) * 2,
            .unsigned_flag = 0,
            .default_stride = false};
        __ppl_tensor_info A_shared_4 = {.shape = {1, 128, 1, 128},
                                        .stride = NULL,
                                        .addr = A_shared_3.addr,
                                        .dtype = DT_FP16,
                                        .mode = 0,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
        tpu_gdma_cpy_S2L(
            A_shared_4.addr, A_1.addr, &A_shared_4.shape,
            (A_shared_4.default_stride ? NULL : &A_shared_4.stride),
            (A_1.default_stride ? NULL : &A_1.stride), DT_FP16);
        __ppl_tensor_info condval_1;
        if ((((k + 1) % 2) == 0)) {
          condval_1 = B_shared_0;
        } else {
          condval_1 = B_shared_1;
        }
        __ppl_tensor_info B_shared_3 = condval_1;
        __ppl_tensor_info B_1 = {
            .shape = {1, 128, 1, 128},
            .stride = {617796, 786, 786, 1},
            .addr =
                v5.addr + ((((k * 128) + 128)) * 786 + ((bx * 128)) * 1) * 2,
            .dtype = DT_FP16,
            .mode = 2,
            .size = 1,
            .offset = ((((k * 128) + 128)) * 786 + ((bx * 128)) * 1) * 2,
            .unsigned_flag = 0,
            .default_stride = false};
        __ppl_tensor_info B_shared_4 = {.shape = {1, 128, 1, 128},
                                        .stride = NULL,
                                        .addr = B_shared_3.addr,
                                        .dtype = DT_FP16,
                                        .mode = 0,
                                        .size = 1,
                                        .offset = 0,
                                        .unsigned_flag = 0,
                                        .default_stride = true};
        tpu_gdma_cpy_S2L(
            B_shared_4.addr, B_1.addr, &B_shared_4.shape,
            (B_shared_4.default_stride ? NULL : &B_shared_4.stride),
            (B_1.default_stride ? NULL : &B_1.stride), DT_FP16);
        __ppl_tensor_info condval_2;
        if (((k % 2) == 0)) {
          condval_2 = B_shared_0;
        } else {
          condval_2 = B_shared_1;
        }
        __ppl_tensor_info B_shared_5 = condval_2;
        __ppl_tensor_info condval_3;
        if (((k % 2) == 0)) {
          condval_3 = A_shared_0;
        } else {
          condval_3 = A_shared_1;
        }
        __ppl_tensor_info A_shared_5 = condval_3;
        tpu_bdc_fp_mm(C_shared.addr, A_shared_5.addr, B_shared_5.addr, 128, 128,
                      128, DT_FP32, DT_FP16, true);
        tpu_parallel_end();
      }
      tpu_bdc_fp_mm(C_shared.addr, A_shared_0.addr, B_shared_0.addr, 128, 128,
                    128, DT_FP32, DT_FP16, true);
      __ppl_tensor_info C_shared_1 = {.shape = {1, 128, 1, 128},
                                      .stride = NULL,
                                      .addr = C_shared.addr,
                                      .dtype = DT_FP32,
                                      .mode = 0,
                                      .size = 1,
                                      .offset = 0,
                                      .unsigned_flag = 0,
                                      .default_stride = true};
      __ppl_tensor_info C = {
          .shape = {1, 128, 1, 128},
          .stride = {301824, 786, 786, 1},
          .addr = v6.addr + (((by * 128)) * 786 + ((bx * 128)) * 1) * 4,
          .dtype = DT_FP32,
          .mode = 2,
          .size = 1,
          .offset = (((by * 128)) * 786 + ((bx * 128)) * 1) * 4,
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