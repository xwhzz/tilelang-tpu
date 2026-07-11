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

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4) {
  __ppl_tensor_info v8 = {.shape = {1 ,8192, 1, 128}, .stride = {0}, .addr = v4, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2097152, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {1 ,1, 1, 1}, .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1 ,128, 1, 128}, .stride = {0}, .addr = v2, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 16384, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,8192, 1, 128}, .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2097152, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info right_fp8_0 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 212992, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_fp8_0.stride, 0, &right_fp8_0.shape, DT_FP8E4M3);
  __ppl_tensor_info right_fp8_1 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 196608, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_fp8_1.stride, 0, &right_fp8_1.shape, DT_FP8E4M3);
  __ppl_tensor_info right_bf16_0 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 180224, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_bf16_0.stride, 0, &right_bf16_0.shape, DT_BFP16);
  __ppl_tensor_info right_bf16_1 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 163840, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_bf16_1.stride, 0, &right_bf16_1.shape, DT_BFP16);
  __ppl_tensor_info scale_group_shared = {.shape = { 1, 1, 1, 1}, .stride = {0}, .addr = 204800, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&scale_group_shared.stride, 0, &scale_group_shared.shape, DT_BFP16);
  __ppl_tensor_info scale_kgroup_0 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 245760, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&scale_kgroup_0.stride, 0, &scale_kgroup_0.shape, DT_BFP16);
  __ppl_tensor_info scale_kgroup_1 = {.shape = { 1, 64, 1, 1}, .stride = {0}, .addr = 229376, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&scale_kgroup_1.stride, 0, &scale_kgroup_1.shape, DT_BFP16);
  __ppl_tensor_info right_scaled_0 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 147456, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_scaled_0.stride, 0, &right_scaled_0.shape, DT_BFP16);
  __ppl_tensor_info right_scaled_1 = {.shape = { 1, 64, 1, 128}, .stride = {0}, .addr = 131072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&right_scaled_1.stride, 0, &right_scaled_1.shape, DT_BFP16);
  __ppl_tensor_info G_right_fp8 = {.shape = {1, 64, 1, 128}, .stride = {16384, 128, 128, 1} , .addr = v6.addr + ((0) * 128+(0) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info right_fp8_0_1 = {.shape = {1, 64, 1, 128}, .stride = right_fp8_0.stride, .addr = right_fp8_0.addr + ((0) * right_fp8_0.stride.c+(0) * right_fp8_0.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * right_fp8_0.stride.c+(0) * right_fp8_0.stride.w ) * 1, .unsigned_flag = 0, .default_stride = right_fp8_0.default_stride};
  tpu_gdma_cpy_S2L(right_fp8_0_1.addr, G_right_fp8.addr, &right_fp8_0_1.shape, (right_fp8_0_1.default_stride ? NULL : &right_fp8_0_1.stride), (G_right_fp8.default_stride ? NULL : &G_right_fp8.stride), DT_FP8E4M3);
  __ppl_tensor_info G_right_fp8_1 = {.shape = {1, 64, 1, 128}, .stride = {16384, 128, 128, 1} , .addr = v6.addr + ((64) * 128+(0) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((64) * 128+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info right_fp8_1_1 = {.shape = {1, 64, 1, 128}, .stride = right_fp8_1.stride, .addr = right_fp8_1.addr + ((0) * right_fp8_1.stride.c+(0) * right_fp8_1.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * right_fp8_1.stride.c+(0) * right_fp8_1.stride.w ) * 1, .unsigned_flag = 0, .default_stride = right_fp8_1.default_stride};
  tpu_gdma_cpy_S2L(right_fp8_1_1.addr, G_right_fp8_1.addr, &right_fp8_1_1.shape, (right_fp8_1_1.default_stride ? NULL : &right_fp8_1_1.stride), (G_right_fp8_1.default_stride ? NULL : &G_right_fp8_1.stride), DT_FP8E4M3);
  __ppl_tensor_info right_fp8_0_2 = {.shape = {1, 64, 1, 128}, .stride = right_fp8_0.stride, .addr = right_fp8_0.addr + ((0) * right_fp8_0.stride.c+(0) * right_fp8_0.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * right_fp8_0.stride.c+(0) * right_fp8_0.stride.w ) * 1, .unsigned_flag = 0, .default_stride = right_fp8_0.default_stride};
  __ppl_tensor_info right_bf16_0_1 = {.shape = {1, 64, 1, 128}, .stride = right_bf16_0.stride, .addr = right_bf16_0.addr + ((0) * right_bf16_0.stride.c+(0) * right_bf16_0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * right_bf16_0.stride.c+(0) * right_bf16_0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = right_bf16_0.default_stride};
  tpu_bdc_cast(right_bf16_0_1.addr, right_fp8_0_2.addr, &right_bf16_0_1.shape, (right_bf16_0_1.default_stride ? NULL : &right_bf16_0_1.stride), (right_fp8_0_2.default_stride ? NULL : &right_fp8_0_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
  __ppl_tensor_info right_fp8_1_2 = {.shape = {1, 64, 1, 128}, .stride = right_fp8_1.stride, .addr = right_fp8_1.addr + ((0) * right_fp8_1.stride.c+(0) * right_fp8_1.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * right_fp8_1.stride.c+(0) * right_fp8_1.stride.w ) * 1, .unsigned_flag = 0, .default_stride = right_fp8_1.default_stride};
  __ppl_tensor_info right_bf16_1_1 = {.shape = {1, 64, 1, 128}, .stride = right_bf16_1.stride, .addr = right_bf16_1.addr + ((0) * right_bf16_1.stride.c+(0) * right_bf16_1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * right_bf16_1.stride.c+(0) * right_bf16_1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = right_bf16_1.default_stride};
  tpu_bdc_cast(right_bf16_1_1.addr, right_fp8_1_2.addr, &right_bf16_1_1.shape, (right_bf16_1_1.default_stride ? NULL : &right_bf16_1_1.stride), (right_fp8_1_2.default_stride ? NULL : &right_fp8_1_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
  __ppl_tensor_info G_scale_group = {.shape = {1, 1, 1, 1}, .stride = {1, 1, 1, 1} , .addr = v7.addr + ((0) * 1+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info scale_group_shared_1 = {.shape = {1, 1, 1, 1}, .stride = scale_group_shared.stride, .addr = scale_group_shared.addr + ((0) * scale_group_shared.stride.c+(0) * scale_group_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_group_shared.stride.c+(0) * scale_group_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_group_shared.default_stride};
  tpu_gdma_cpy_S2L(scale_group_shared_1.addr, G_scale_group.addr, &scale_group_shared_1.shape, (scale_group_shared_1.default_stride ? NULL : &scale_group_shared_1.stride), (G_scale_group.default_stride ? NULL : &G_scale_group.stride), DT_BFP16);
  if (scale_group_shared.size) {
    tpu_bdc_npu_bcast(scale_kgroup_0.addr, scale_group_shared.addr, &scale_kgroup_0.shape, DT_BFP16);
  }
  dim4 scale_kgroup_0_stride;
  tpu_aligned_stride(&scale_kgroup_0_stride, 0, &scale_kgroup_0.shape, DT_BFP16);
  scale_kgroup_0_stride.w = 0;
  tpu_bdc_fp_mul( right_scaled_0.addr, right_bf16_0.addr, scale_kgroup_0.addr, &right_scaled_0.shape, (right_scaled_0.default_stride ? NULL : &right_scaled_0.stride), (right_bf16_0.default_stride ? NULL : &right_bf16_0.stride), &scale_kgroup_0_stride, DT_BFP16);
  if (scale_group_shared.size) {
    tpu_bdc_npu_bcast(scale_kgroup_1.addr, scale_group_shared.addr, &scale_kgroup_1.shape, DT_BFP16);
  }
  dim4 scale_kgroup_1_stride;
  tpu_aligned_stride(&scale_kgroup_1_stride, 0, &scale_kgroup_1.shape, DT_BFP16);
  scale_kgroup_1_stride.w = 0;
  tpu_bdc_fp_mul( right_scaled_1.addr, right_bf16_1.addr, scale_kgroup_1.addr, &right_scaled_1.shape, (right_scaled_1.default_stride ? NULL : &right_scaled_1.stride), (right_bf16_1.default_stride ? NULL : &right_bf16_1.stride), &scale_kgroup_1_stride, DT_BFP16);
  int m_blk_core_idx = tpu_workitem_index();
  int m_blk_core_num = tpu_workitem_num();
  if (m_blk_core_num <= 0) m_blk_core_num = 1;
  for (int m_blk = m_blk_core_idx; m_blk < 64; m_blk += m_blk_core_num) {
    __ppl_tensor_info left_shared = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 65536, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&left_shared.stride, 0, &left_shared.shape, DT_BFP16);
    __ppl_tensor_info out_shared_0 = {.shape = { 1, 128, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_shared_0.stride, 0, &out_shared_0.shape, DT_FP32);
    __ppl_tensor_info out_shared_1 = {.shape = { 1, 128, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_shared_1.stride, 0, &out_shared_1.shape, DT_FP32);
    __ppl_tensor_info out_cast_0 = {.shape = { 1, 128, 1, 64}, .stride = {0}, .addr = 114688, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_0.stride, 0, &out_cast_0.shape, DT_BFP16);
    __ppl_tensor_info out_cast_1 = {.shape = { 1, 128, 1, 64}, .stride = {0}, .addr = 98304, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast_1.stride, 0, &out_cast_1.shape, DT_BFP16);
    __ppl_tensor_info G_left = {.shape = {1, 128, 1, 128}, .stride = {1048576, 128, 128, 1} , .addr = v5.addr + (((m_blk * 128)) * 128+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((m_blk * 128)) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info left_shared_1 = {.shape = {1, 128, 1, 128}, .stride = left_shared.stride, .addr = left_shared.addr + ((0) * left_shared.stride.c+(0) * left_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * left_shared.stride.c+(0) * left_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = left_shared.default_stride};
    tpu_gdma_cpy_S2L(left_shared_1.addr, G_left.addr, &left_shared_1.shape, (left_shared_1.default_stride ? NULL : &left_shared_1.stride), (G_left.default_stride ? NULL : &G_left.stride), DT_BFP16);
    {
    scalar_t out_shared_0_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_shared_0.addr, out_shared_0_scalar_f32, &out_shared_0.shape, (out_shared_0.default_stride ? NULL : &out_shared_0.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(out_shared_0.addr, left_shared.addr, right_scaled_0.addr, 128, 128, 64, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_shared_0_1 = {.shape = {1, 128, 1, 64}, .stride = out_shared_0.stride, .addr = out_shared_0.addr + ((0) * out_shared_0.stride.c+(0) * out_shared_0.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_shared_0.stride.c+(0) * out_shared_0.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_shared_0.default_stride};
    __ppl_tensor_info out_cast_0_1 = {.shape = {1, 128, 1, 64}, .stride = out_cast_0.stride, .addr = out_cast_0.addr + ((0) * out_cast_0.stride.c+(0) * out_cast_0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_0.stride.c+(0) * out_cast_0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_0.default_stride};
    tpu_bdc_cast(out_cast_0_1.addr, out_shared_0_1.addr, &out_cast_0_1.shape, (out_cast_0_1.default_stride ? NULL : &out_cast_0_1.stride), (out_shared_0_1.default_stride ? NULL : &out_shared_0_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_0_2 = {.shape = {1, 128, 1, 64}, .stride = out_cast_0.stride, .addr = out_cast_0.addr + ((0) * out_cast_0.stride.c+(0) * out_cast_0.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_0.stride.c+(0) * out_cast_0.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_0.default_stride};
    __ppl_tensor_info G_out = {.shape = {1, 128, 1, 64}, .stride = {1048576, 128, 128, 1} , .addr = v8.addr + (((m_blk * 128)) * 128+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((m_blk * 128)) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(G_out.addr, out_cast_0_2.addr, &G_out.shape, (G_out.default_stride ? NULL : &G_out.stride), (out_cast_0_2.default_stride ? NULL : &out_cast_0_2.stride), DT_BFP16);
    {
    scalar_t out_shared_1_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_shared_1.addr, out_shared_1_scalar_f32, &out_shared_1.shape, (out_shared_1.default_stride ? NULL : &out_shared_1.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(out_shared_1.addr, left_shared.addr, right_scaled_1.addr, 128, 128, 64, DT_FP32, DT_BFP16);
    __ppl_tensor_info out_shared_1_1 = {.shape = {1, 128, 1, 64}, .stride = out_shared_1.stride, .addr = out_shared_1.addr + ((0) * out_shared_1.stride.c+(0) * out_shared_1.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_shared_1.stride.c+(0) * out_shared_1.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_shared_1.default_stride};
    __ppl_tensor_info out_cast_1_1 = {.shape = {1, 128, 1, 64}, .stride = out_cast_1.stride, .addr = out_cast_1.addr + ((0) * out_cast_1.stride.c+(0) * out_cast_1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_1.stride.c+(0) * out_cast_1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_1.default_stride};
    tpu_bdc_cast(out_cast_1_1.addr, out_shared_1_1.addr, &out_cast_1_1.shape, (out_cast_1_1.default_stride ? NULL : &out_cast_1_1.stride), (out_shared_1_1.default_stride ? NULL : &out_shared_1_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_1_2 = {.shape = {1, 128, 1, 64}, .stride = out_cast_1.stride, .addr = out_cast_1.addr + ((0) * out_cast_1.stride.c+(0) * out_cast_1.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast_1.stride.c+(0) * out_cast_1.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast_1.default_stride};
    __ppl_tensor_info G_out_1 = {.shape = {1, 128, 1, 64}, .stride = {1048576, 128, 128, 1} , .addr = v8.addr + (((m_blk * 128)) * 128+(64) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((m_blk * 128)) * 128+(64) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(G_out_1.addr, out_cast_1_2.addr, &G_out_1.shape, (G_out_1.default_stride ? NULL : &G_out_1.stride), (out_cast_1_2.default_stride ? NULL : &out_cast_1_2.stride), DT_BFP16);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  main_kernel_inner(api->v1,
    api->v2,
    api->v3,
    api->v4);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
