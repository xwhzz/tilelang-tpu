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

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11, global_addr_t v12, global_addr_t v13) {
  __ppl_tensor_info v26 = {.shape = {1 ,1, 1, 128}, .stride = {0}, .addr = v13, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v24 = {.shape = {}, .stride = {0}, .addr = v11, .dtype = DT_INT32, .mode = 2, .align_mode = 0, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v23 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v10, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v22 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v9, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v8, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v7, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v6, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v25 = {.shape = {}, .stride = {0}, .addr = v12, .dtype = DT_INT32, .mode = 2, .align_mode = 0, .size = 4, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v5, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v17 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v4, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v16 = {.shape = {1 ,128, 1, 1}, .stride = {0}, .addr = v3, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 128, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v15 = {.shape = {1 ,128, 1, 64}, .stride = {0}, .addr = v2, .dtype = DT_UINT8, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v14 = {.shape = {1 ,1, 1, 128}, .stride = {0}, .addr = v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 256, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info output_acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 17536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&output_acc.stride, 0, &output_acc.shape, DT_FP32);
  __ppl_tensor_info output_cast = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&output_cast.stride, 0, &output_cast.shape, DT_FP16);
  {
  scalar_t output_acc_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(output_acc.addr, output_acc_scalar_f32, &output_acc.shape, (output_acc.default_stride ? NULL : &output_acc.stride), DT_FP32);
  }
  __ppl_tensor_info input_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 17280, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&input_shared.stride, 0, &input_shared.shape, DT_FP16);
  __ppl_tensor_info w_shared = {.shape = { 1, 128, 1, 64}, .stride = {0}, .addr = 17152, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&w_shared.stride, 0, &w_shared.shape, DT_UINT8);
  __ppl_tensor_info zp_shared = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 33280, .dtype = DT_UINT8, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&zp_shared.stride, 0, &zp_shared.shape, DT_UINT8);
  __ppl_tensor_info scale_shared = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 49664, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&scale_shared.stride, 0, &scale_shared.shape, DT_FP16);
  __ppl_tensor_info dequant = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&dequant.stride, 0, &dequant.shape, DT_FP16);
  __ppl_tensor_info dequant_work = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 65536, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&dequant_work.stride, 0, &dequant_work.shape, DT_FP16);
  __ppl_tensor_info gate_out = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 18048, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&gate_out.stride, 0, &gate_out.shape, DT_FP32);
  __ppl_tensor_info up_out = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 18560, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&up_out.stride, 0, &up_out.shape, DT_FP32);
  __ppl_tensor_info gate_exp = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&gate_exp.stride, 0, &gate_exp.shape, DT_FP32);
  __ppl_tensor_info gate_denom = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&gate_denom.stride, 0, &gate_denom.shape, DT_FP32);
  __ppl_tensor_info swiglu_out = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&swiglu_out.stride, 0, &swiglu_out.shape, DT_FP32);
  __ppl_tensor_info work0 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&work0.stride, 0, &work0.shape, DT_FP32);
  __ppl_tensor_info work1 = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 8, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&work1.stride, 0, &work1.shape, DT_FP32);
  __ppl_tensor_info coeff = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 65664, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&coeff.stride, 0, &coeff.shape, DT_FP32);
  __ppl_tensor_info table = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&table.stride, 0, &table.shape, DT_FP32);
  {
  scalar_t gate_out_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(gate_out.addr, gate_out_scalar_f32, &gate_out.shape, (gate_out.default_stride ? NULL : &gate_out.stride), DT_FP32);
  }
  {
  scalar_t up_out_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(up_out.addr, up_out_scalar_f32, &up_out.shape, (up_out.default_stride ? NULL : &up_out.stride), DT_FP32);
  }
  __ppl_tensor_info input_tensor = {.shape = {1, 1, 1, 128}, .stride = {128, 128, 128, 1} , .addr = v14.addr + ((0) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info input_shared_1 = {.shape = {1, 1, 1, 128}, .stride = input_shared.stride, .addr = input_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = input_shared.default_stride};
  tpu_gdma_cpy_S2L(input_shared_1.addr, input_tensor.addr, &input_shared_1.shape, (input_shared_1.default_stride ? NULL : &input_shared_1.stride), (input_tensor.default_stride ? NULL : &input_tensor.stride), DT_FP16);
  __ppl_tensor_info gate_w = {.shape = {1, 128, 1, 64}, .stride = {8192, 64, 64, 1} , .addr = v18.addr + ((0) * 64+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info w_shared_1 = {.shape = {1, 128, 1, 64}, .stride = w_shared.stride, .addr = w_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = w_shared.default_stride};
  tpu_gdma_cpy_S2L(w_shared_1.addr, gate_w.addr, &w_shared_1.shape, (w_shared_1.default_stride ? NULL : &w_shared_1.stride), (gate_w.default_stride ? NULL : &gate_w.stride), DT_UINT8);
  __ppl_tensor_info gate_zp = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v19.addr + ((0) * 1+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info zp_shared_1 = {.shape = {1, 128, 1, 1}, .stride = zp_shared.stride, .addr = zp_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = zp_shared.default_stride};
  tpu_gdma_cpy_S2L(zp_shared_1.addr, gate_zp.addr, &zp_shared_1.shape, (zp_shared_1.default_stride ? NULL : &zp_shared_1.stride), (gate_zp.default_stride ? NULL : &gate_zp.stride), DT_UINT8);
  __ppl_tensor_info gate_scale = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v20.addr + ((0) * 1+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info scale_shared_1 = {.shape = {1, 128, 1, 1}, .stride = scale_shared.stride, .addr = scale_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = scale_shared.default_stride};
  tpu_gdma_cpy_S2L(scale_shared_1.addr, gate_scale.addr, &scale_shared_1.shape, (scale_shared_1.default_stride ? NULL : &scale_shared_1.stride), (gate_scale.default_stride ? NULL : &gate_scale.stride), DT_FP16);
  {
    tpu_bdc_cast(dequant.addr, w_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (w_shared.default_stride ? NULL : &w_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    tpu_bdc_cast(dequant_work.addr, zp_shared.addr, &dequant_work.shape, (dequant_work.default_stride ? NULL : &dequant_work.stride), (zp_shared.default_stride ? NULL : &zp_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    dim4 dequant_work_bc_stride;
    tpu_aligned_stride(&dequant_work_bc_stride, 0, &dequant_work.shape, DT_FP16);
    dequant_work_bc_stride.w = 0;
    tpu_bdc_fp_sub(dequant.addr, dequant.addr, dequant_work.addr + ((0 % 2) * 2), &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &dequant_work_bc_stride, DT_FP16);
    dim4 scale_shared_bc_stride;
    tpu_aligned_stride(&scale_shared_bc_stride, 0, &scale_shared.shape, DT_FP16);
    scale_shared_bc_stride.w = 0;
    tpu_bdc_fp_mul(dequant.addr, dequant.addr, scale_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &scale_shared_bc_stride, DT_FP16);
  }
  tpu_bdc_fp_mm_R_trans(gate_out.addr, input_shared.addr, dequant.addr, 1, 128, 128, DT_FP32, DT_FP16);
  __ppl_tensor_info up_w = {.shape = {1, 128, 1, 64}, .stride = {8192, 64, 64, 1} , .addr = v15.addr + ((0) * 64+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info w_shared_2 = {.shape = {1, 128, 1, 64}, .stride = w_shared.stride, .addr = w_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = w_shared.default_stride};
  tpu_gdma_cpy_S2L(w_shared_2.addr, up_w.addr, &w_shared_2.shape, (w_shared_2.default_stride ? NULL : &w_shared_2.stride), (up_w.default_stride ? NULL : &up_w.stride), DT_UINT8);
  __ppl_tensor_info up_zp = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v16.addr + ((0) * 1+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info zp_shared_2 = {.shape = {1, 128, 1, 1}, .stride = zp_shared.stride, .addr = zp_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = zp_shared.default_stride};
  tpu_gdma_cpy_S2L(zp_shared_2.addr, up_zp.addr, &zp_shared_2.shape, (zp_shared_2.default_stride ? NULL : &zp_shared_2.stride), (up_zp.default_stride ? NULL : &up_zp.stride), DT_UINT8);
  __ppl_tensor_info up_scale = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v17.addr + ((0) * 1+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info scale_shared_2 = {.shape = {1, 128, 1, 1}, .stride = scale_shared.stride, .addr = scale_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = scale_shared.default_stride};
  tpu_gdma_cpy_S2L(scale_shared_2.addr, up_scale.addr, &scale_shared_2.shape, (scale_shared_2.default_stride ? NULL : &scale_shared_2.stride), (up_scale.default_stride ? NULL : &up_scale.stride), DT_FP16);
  {
    tpu_bdc_cast(dequant.addr, w_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (w_shared.default_stride ? NULL : &w_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    tpu_bdc_cast(dequant_work.addr, zp_shared.addr, &dequant_work.shape, (dequant_work.default_stride ? NULL : &dequant_work.stride), (zp_shared.default_stride ? NULL : &zp_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    dim4 dequant_work_bc_stride;
    tpu_aligned_stride(&dequant_work_bc_stride, 0, &dequant_work.shape, DT_FP16);
    dequant_work_bc_stride.w = 0;
    tpu_bdc_fp_sub(dequant.addr, dequant.addr, dequant_work.addr + ((0 % 2) * 2), &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &dequant_work_bc_stride, DT_FP16);
    dim4 scale_shared_bc_stride;
    tpu_aligned_stride(&scale_shared_bc_stride, 0, &scale_shared.shape, DT_FP16);
    scale_shared_bc_stride.w = 0;
    tpu_bdc_fp_mul(dequant.addr, dequant.addr, scale_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &scale_shared_bc_stride, DT_FP16);
  }
  tpu_bdc_fp_mm_R_trans(up_out.addr, input_shared.addr, dequant.addr, 1, 128, 128, DT_FP32, DT_FP16);
  tpu_bdc_fp_mul_C( gate_exp.addr, gate_out.addr, (scalar_t){.f32 = -1}, &gate_exp.shape, (gate_exp.default_stride ? NULL : &gate_exp.stride), (gate_out.default_stride ? NULL : &gate_out.stride), DT_FP32);
  tpu_bdc_load_fp32_exp_coeff(coeff.addr);
  tpu_bdc_load_fp32_exp_table(table.addr);
  tpu_bdc_fp32_exp(gate_exp.addr, gate_exp.addr, work0.addr, work1.addr, coeff.addr, table.addr, &gate_exp.shape);
  tpu_bdc_fp_add_C( gate_denom.addr, gate_exp.addr, (scalar_t){.f32 = 1}, &gate_denom.shape, (gate_denom.default_stride ? NULL : &gate_denom.stride), (gate_exp.default_stride ? NULL : &gate_exp.stride), DT_FP32);
  tpu_bdc_fp_div( gate_exp.addr, gate_out.addr, gate_denom.addr, &gate_exp.shape, (gate_exp.default_stride ? NULL : &gate_exp.stride), (gate_out.default_stride ? NULL : &gate_out.stride), (gate_denom.default_stride ? NULL : &gate_denom.stride), DT_FP32);
  tpu_bdc_fp_mul( swiglu_out.addr, up_out.addr, gate_exp.addr, &swiglu_out.shape, (swiglu_out.default_stride ? NULL : &swiglu_out.stride), (up_out.default_stride ? NULL : &up_out.stride), (gate_exp.default_stride ? NULL : &gate_exp.stride), DT_FP32);
  __ppl_tensor_info down_w = {.shape = {1, 128, 1, 64}, .stride = {8192, 64, 64, 1} , .addr = v21.addr + ((0) * 64+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info w_shared_3 = {.shape = {1, 128, 1, 64}, .stride = w_shared.stride, .addr = w_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = w_shared.default_stride};
  tpu_gdma_cpy_S2L(w_shared_3.addr, down_w.addr, &w_shared_3.shape, (w_shared_3.default_stride ? NULL : &w_shared_3.stride), (down_w.default_stride ? NULL : &down_w.stride), DT_UINT8);
  __ppl_tensor_info down_zp = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v22.addr + ((0) * 1+(0) * 1 ) * 1, .dtype = DT_UINT8, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info zp_shared_3 = {.shape = {1, 128, 1, 1}, .stride = zp_shared.stride, .addr = zp_shared.addr, .dtype = DT_UINT8, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = zp_shared.default_stride};
  tpu_gdma_cpy_S2L(zp_shared_3.addr, down_zp.addr, &zp_shared_3.shape, (zp_shared_3.default_stride ? NULL : &zp_shared_3.stride), (down_zp.default_stride ? NULL : &down_zp.stride), DT_UINT8);
  __ppl_tensor_info down_scale = {.shape = {1, 128, 1, 1}, .stride = {128, 1, 1, 1} , .addr = v23.addr + ((0) * 1+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info scale_shared_3 = {.shape = {1, 128, 1, 1}, .stride = scale_shared.stride, .addr = scale_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = scale_shared.default_stride};
  tpu_gdma_cpy_S2L(scale_shared_3.addr, down_scale.addr, &scale_shared_3.shape, (scale_shared_3.default_stride ? NULL : &scale_shared_3.stride), (down_scale.default_stride ? NULL : &down_scale.stride), DT_FP16);
  {
    tpu_bdc_cast(dequant.addr, w_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (w_shared.default_stride ? NULL : &w_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    tpu_bdc_cast(dequant_work.addr, zp_shared.addr, &dequant_work.shape, (dequant_work.default_stride ? NULL : &dequant_work.stride), (zp_shared.default_stride ? NULL : &zp_shared.stride), DT_FP16, DT_UINT4, RM_HALF_TO_EVEN);
    dim4 dequant_work_bc_stride;
    tpu_aligned_stride(&dequant_work_bc_stride, 0, &dequant_work.shape, DT_FP16);
    dequant_work_bc_stride.w = 0;
    tpu_bdc_fp_sub(dequant.addr, dequant.addr, dequant_work.addr + ((0 % 2) * 2), &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &dequant_work_bc_stride, DT_FP16);
    dim4 scale_shared_bc_stride;
    tpu_aligned_stride(&scale_shared_bc_stride, 0, &scale_shared.shape, DT_FP16);
    scale_shared_bc_stride.w = 0;
    tpu_bdc_fp_mul(dequant.addr, dequant.addr, scale_shared.addr, &dequant.shape, (dequant.default_stride ? NULL : &dequant.stride), (dequant.default_stride ? NULL : &dequant.stride), &scale_shared_bc_stride, DT_FP16);
  }
  __ppl_tensor_info swiglu_out_1 = {.shape = {1, 1, 1, 128}, .stride = swiglu_out.stride, .addr = swiglu_out.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = swiglu_out.default_stride};
  __ppl_tensor_info input_shared_2 = {.shape = {1, 1, 1, 128}, .stride = input_shared.stride, .addr = input_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = input_shared.default_stride};
  tpu_bdc_cast(input_shared_2.addr, swiglu_out_1.addr, &input_shared_2.shape, (input_shared_2.default_stride ? NULL : &input_shared_2.stride), (swiglu_out_1.default_stride ? NULL : &swiglu_out_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
  tpu_bdc_fp_mm_R_trans(output_acc.addr, input_shared.addr, dequant.addr, 1, 128, 128, DT_FP32, DT_FP16);
  __ppl_tensor_info output_acc_1 = {.shape = {1, 1, 1, 128}, .stride = output_acc.stride, .addr = output_acc.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = output_acc.default_stride};
  __ppl_tensor_info output_cast_1 = {.shape = {1, 1, 1, 128}, .stride = output_cast.stride, .addr = output_cast.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = output_cast.default_stride};
  tpu_bdc_cast(output_cast_1.addr, output_acc_1.addr, &output_cast_1.shape, (output_cast_1.default_stride ? NULL : &output_cast_1.stride), (output_acc_1.default_stride ? NULL : &output_acc_1.stride), DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
  __ppl_tensor_info output_cast_2 = {.shape = {1, 1, 1, 128}, .stride = output_cast.stride, .addr = output_cast.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = output_cast.default_stride};
  __ppl_tensor_info output_tensor = {.shape = {1, 1, 1, 128}, .stride = {128, 128, 128, 1} , .addr = v26.addr + ((0) * 128+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  tpu_gdma_cpy_L2S(output_tensor.addr, output_cast_2.addr, &output_tensor.shape, (output_tensor.default_stride ? NULL : &output_tensor.stride), (output_cast_2.default_stride ? NULL : &output_cast_2.stride), DT_FP16);
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
  global_addr_t v5;
  global_addr_t v6;
  global_addr_t v7;
  global_addr_t v8;
  global_addr_t v9;
  global_addr_t v10;
  global_addr_t v11;
  global_addr_t v12;
  global_addr_t v13;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  main_kernel_inner(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5,
    api->v6,
    api->v7,
    api->v8,
    api->v9,
    api->v10,
    api->v11,
    api->v12,
    api->v13);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
