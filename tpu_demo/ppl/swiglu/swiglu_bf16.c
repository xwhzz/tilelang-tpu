#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t v1, global_addr_t v2, global_addr_t v3) {
  __ppl_tensor_info v6 = {.shape = {1 ,1, 1, 32}, .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,1, 1, 32}, .stride = {0}, .addr = v2, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1 ,1, 1, 32}, .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_ori = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 229376, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info right_ori = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 212992, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 163840, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info right = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 147456, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 131072, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg_exp = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 114688, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info ones = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 98304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg_exp_1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_neg_exp_1_div = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info out = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info out_ori = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 196608, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work0_1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info work1_1 = {.shape = { 1, 64, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info coeff_1 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 180224, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info table_1 = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = true};
  scalar_t ones_scalar_f32 = {.f32 = 1};
  tpu_bdc_set_C(ones.addr, ones_scalar_f32, &ones.shape, (ones.default_stride ? NULL : &ones.stride), DT_FP32);
  __ppl_tensor_info G_in = {.shape = {1, 64, 1, 64}, .stride = {32, 32, 32, 1} , .addr = v5.addr + ((0) * 32+(0) * 32 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 32+(0) * 32 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info x_ori_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = x_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_gdma_cpy_S2L(x_ori_1.addr, G_in.addr, &x_ori_1.shape, (x_ori_1.default_stride ? NULL : &x_ori_1.stride), (G_in.default_stride ? NULL : &G_in.stride), DT_BFP16);
  __ppl_tensor_info G_right = {.shape = {1, 64, 1, 64}, .stride = {32, 32, 32, 1} , .addr = v6.addr + ((0) * 32+(0) * 32 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 32+(0) * 32 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info right_ori_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = right_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_gdma_cpy_S2L(right_ori_1.addr, G_right.addr, &right_ori_1.shape, (right_ori_1.default_stride ? NULL : &right_ori_1.stride), (G_right.default_stride ? NULL : &G_right.stride), DT_BFP16);
  __ppl_tensor_info x_ori_2 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = x_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info x_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = x.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_bdc_cast(x_1.addr, x_ori_2.addr, &x_1.shape, (x_1.default_stride ? NULL : &x_1.stride), (x_ori_2.default_stride ? NULL : &x_ori_2.stride), DT_FP32, DT_BFP16, RM_HALF_TO_EVEN);
  __ppl_tensor_info right_ori_2 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = right_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info right_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = right.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_bdc_cast(right_1.addr, right_ori_2.addr, &right_1.shape, (right_1.default_stride ? NULL : &right_1.stride), (right_ori_2.default_stride ? NULL : &right_ori_2.stride), DT_FP32, DT_BFP16, RM_HALF_TO_EVEN);
  tpu_bdc_fp_mul_C( x_neg.addr, x.addr, (scalar_t){.f32 = -1}, &x_neg.shape, (x_neg.default_stride ? NULL : &x_neg.stride), (x_neg.default_stride ? NULL : &x_neg.stride), DT_FP32);
  tpu_bdc_load_fp32_exp_coeff(coeff_1.addr);
  tpu_bdc_load_fp32_exp_table(table_1.addr);
  tpu_bdc_fp32_exp(x_neg_exp.addr, x_neg_exp.addr, work0_1.addr, work1_1.addr, coeff_1.addr, table_1.addr, &x_neg_exp.shape);
  tpu_bdc_fp_add( x_neg_exp_1.addr, x_neg_exp.addr, ones.addr, &x_neg_exp_1.shape, (x_neg_exp_1.default_stride ? NULL : &x_neg_exp_1.stride), (x_neg_exp.default_stride ? NULL : &x_neg_exp.stride), (ones.default_stride ? NULL : &ones.stride), DT_FP32);
  tpu_bdc_fp_div( x_neg_exp_1_div.addr, x.addr, x_neg_exp_1.addr, &x_neg_exp_1_div.shape, (x_neg_exp_1_div.default_stride ? NULL : &x_neg_exp_1_div.stride), (x.default_stride ? NULL : &x.stride), (x_neg_exp_1.default_stride ? NULL : &x_neg_exp_1.stride), DT_FP32);
  tpu_bdc_fp_mul( out.addr, right.addr, x_neg_exp_1_div.addr, &out.shape, (out.default_stride ? NULL : &out.stride), (right.default_stride ? NULL : &right.stride), (x_neg_exp_1_div.default_stride ? NULL : &x_neg_exp_1_div.stride), DT_FP32);
  __ppl_tensor_info out_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = out.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info out_ori_1 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = out_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_bdc_cast(out_ori_1.addr, out_1.addr, &out_ori_1.shape, (out_ori_1.default_stride ? NULL : &out_ori_1.stride), (out_1.default_stride ? NULL : &out_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
  __ppl_tensor_info out_ori_2 = {.shape = {1, 64, 1, 64}, .stride = {0}, .addr = out_ori.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info G_out = {.shape = {1, 64, 1, 64}, .stride = {32, 32, 32, 1} , .addr = v4.addr + ((0) * 32+(0) * 32 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 32+(0) * 32 ) * 2, .unsigned_flag = 0, .default_stride = false};
  tpu_gdma_cpy_L2S(G_out.addr, out_ori_2.addr, &G_out.shape, (G_out.default_stride ? NULL : &G_out.stride), (out_ori_2.default_stride ? NULL : &out_ori_2.stride), DT_BFP16);
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
} tpu_kernel_api_main_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
  main(api->v1,
    api->v2,
    api->v3);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)