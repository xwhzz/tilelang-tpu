# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import tilelang
import tilelang.language as T


def matmul(M, N, block_M, block_N, stage, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
    ):
      with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_cpu=True) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), accum_dtype)

        T.ppl_fill(A_shared, T.float32(0))

        T.copy(A_shared, A[by * block_M, bx * block_N])

    return main


# func =  matmul(4096, 8192, 1024, 1024, 512, 128, 2)
func =  matmul(T.symbolic("m"), T.symbolic("n"), 128, 128, 2)
mod = tilelang.lower(func)


# for mm in range(64,4097,64):
#     for nn in range(64, 1025, 64):
#         for kk in range(64, 8193, 64):
#             for stages in [0, 2, 3, 4]:
#               if 4096 % mm != 0 or 8192 % nn != 0 or 1024 % kk != 0:
#                   continue
#               func = matmul(4096, 8192, 1024, mm, nn, kk, stages)
#               mod: str = tilelang.lower(func)
#               if mod.count(".addr = 0") > 1:
#                   print(f"{mm}_{nn}_{kk} configure failed!")
#                   continue
#               mod = mod.replace("""#include "ppl_helper.h"
# static data_type_t __ppl_get_dtype(int type) {
#   data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
#     DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
#     DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
#     DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
#   return __dtype[type];
# }

# void main(global_addr_t v1, global_addr_t v2, global_addr_t v3) {""", """#include "ppl_helper.h"

# static data_type_t __ppl_get_dtype(int type) {
#   data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
#       DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
#       DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
#       DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
#   return __dtype[type];
# }

# typedef struct {
#   dim4 shape;
#   dim4 stride;
#   global_addr_t addr;
#   data_type_t dtype;
#   int mode;
#   int align_mode;
#   int size;
#   int offset;
#   bool unsigned_flag;
#   bool default_stride;
# } __ppl_tensor_info;
# typedef struct {
#   global_addr_t ptr_left_v1;
#   global_addr_t ptr_right_v2;
#   global_addr_t ptr_res_v3;
# } tpu_kernel_api_mm2_fp16_t;
# void mm2_fp16_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3) {""")
#               mod = mod.replace("""typedef struct {
#   global_addr_t v1;
#   global_addr_t v2;
#   global_addr_t v3;
# } tpu_kernel_api_main_args_t;
# void main_kernel(const void * args) {
#   tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
#   main(api->v1,
#     api->v2,
#     api->v3);
#   tpu_poll();
# }
# TPUKERNEL_FUNC_REGISTER(main_kernel)""", """int mm2_fp16(const void * args) {
#   tpu_kernel_api_mm2_fp16_t *api = (tpu_kernel_api_mm2_fp16_t*)args;
#   tpu_initialize();
#   mm2_fp16_inner(api->ptr_left_v1,
#     api->ptr_right_v2,
#     api->ptr_res_v3);
#   tpu_poll();
#   return 0;
# }
# TPUKERNEL_FUNC_REGISTER(mm2_fp16)""")

#               with open(f"kernel_a/matmul_{mm}_{nn}_{kk}_{stages}.c", "w") as f:
#                   f.write(mod)

