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

void llama_mlp_forward(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11) {
  __ppl_tensor_info v22 = {.shape = {}, .stride = {0}, .addr = v11, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {}, .stride = {0}, .addr = v8, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {}, .stride = {0}, .addr = v9, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {}, .stride = {0}, .addr = v7, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v16 = {.shape = {}, .stride = {0}, .addr = v5, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v15 = {.shape = {1 ,16, 1, 16}, .stride = {0}, .addr = v4, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 512, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v14 = {.shape = {1 ,16, 1, 16}, .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 512, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1 ,16, 1, 16}, .stride = {0}, .addr = v10, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 512, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v17 = {.shape = {}, .stride = {0}, .addr = v6, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v13 = {.shape = {1 ,16, 1, 16}, .stride = {0}, .addr = v2, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 512, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v12 = {.shape = {1 ,16, 1, 16}, .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 512, .unsigned_flag = 0, .default_stride = true};
  for (int bx = 0; bx < 2; ++bx) {
    for (int by = 0; by < 2; ++by) {
      __ppl_tensor_info down_out_acc = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 245760, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&down_out_acc.stride, 0, &down_out_acc.shape, DT_FP32);
      __ppl_tensor_info down_out_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 131328, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&down_out_block.stride, 0, &down_out_block.shape, DT_BFP16);
      __ppl_tensor_info ones = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 229376, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ones.stride, 0, &ones.shape, DT_FP32);
      {
      scalar_t ones_scalar_f32 = {.f32 = 1.000000e+00f};
      tpu_bdc_set_C(ones.addr, ones_scalar_f32, &ones.shape, (ones.default_stride ? NULL : &ones.stride), DT_FP32);
      }
      {
      scalar_t down_out_acc_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(down_out_acc.addr, down_out_acc_scalar_f32, &down_out_acc.shape, (down_out_acc.default_stride ? NULL : &down_out_acc.stride), DT_FP32);
      }
      for (int bz = 0; bz < 2; ++bz) {
        __ppl_tensor_info down_weight_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 114944, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&down_weight_block.stride, 0, &down_weight_block.shape, DT_BFP16);
        __ppl_tensor_info gated_up_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 98560, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gated_up_block.stride, 0, &gated_up_block.shape, DT_BFP16);
        __ppl_tensor_info down_out_part_fp32 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 212992, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&down_out_part_fp32.stride, 0, &down_out_part_fp32.shape, DT_FP32);
        __ppl_tensor_info gate_out_block_fp32 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 196608, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gate_out_block_fp32.stride, 0, &gate_out_block_fp32.shape, DT_FP32);
        __ppl_tensor_info up_out_block_fp32 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 180224, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&up_out_block_fp32.stride, 0, &up_out_block_fp32.shape, DT_FP32);
        __ppl_tensor_info silu_out_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 163840, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&silu_out_block.stride, 0, &silu_out_block.shape, DT_FP32);
        __ppl_tensor_info gated_up_fp32 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 147456, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&gated_up_fp32.stride, 0, &gated_up_fp32.shape, DT_FP32);
        __ppl_tensor_info x_neg = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 131072, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&x_neg.stride, 0, &x_neg.shape, DT_FP32);
        __ppl_tensor_info x_neg_exp_1 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 114688, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&x_neg_exp_1.stride, 0, &x_neg_exp_1.shape, DT_FP32);
        __ppl_tensor_info exp_work0 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 98304, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&exp_work0.stride, 0, &exp_work0.shape, DT_FP32);
        __ppl_tensor_info exp_work1 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 81920, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&exp_work1.stride, 0, &exp_work1.shape, DT_FP32);
        __ppl_tensor_info exp_coeff = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&exp_coeff.stride, 0, &exp_coeff.shape, DT_FP32);
        __ppl_tensor_info exp_table = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
        tpu_aligned_stride(&exp_table.stride, 0, &exp_table.shape, DT_FP32);
        {
        scalar_t gate_out_block_fp32_scalar_f32 = {.f32 = 0.000000e+00f};
        tpu_bdc_set_C(gate_out_block_fp32.addr, gate_out_block_fp32_scalar_f32, &gate_out_block_fp32.shape, (gate_out_block_fp32.default_stride ? NULL : &gate_out_block_fp32.stride), DT_FP32);
        }
        {
        scalar_t up_out_block_fp32_scalar_f32 = {.f32 = 0.000000e+00f};
        tpu_bdc_set_C(up_out_block_fp32.addr, up_out_block_fp32_scalar_f32, &up_out_block_fp32.shape, (up_out_block_fp32.default_stride ? NULL : &up_out_block_fp32.stride), DT_FP32);
        }
        for (int kk = 0; kk < 2; ++kk) {
          __ppl_tensor_info x_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 82176, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
          tpu_aligned_stride(&x_block.stride, 0, &x_block.shape, DT_BFP16);
          __ppl_tensor_info gate_weight_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 65792, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
          tpu_aligned_stride(&gate_weight_block.stride, 0, &gate_weight_block.shape, DT_BFP16);
          __ppl_tensor_info up_weight_block = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 57344, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = false};
          tpu_aligned_stride(&up_weight_block.stride, 0, &up_weight_block.shape, DT_BFP16);
          __ppl_tensor_info proj_part_fp32 = {.shape = { 1, 8, 1, 8}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 4, .unsigned_flag = 0, .default_stride = false};
          tpu_aligned_stride(&proj_part_fp32.stride, 0, &proj_part_fp32.shape, DT_FP32);
          __ppl_tensor_info x = {.shape = {1, 8, 1, 8}, .stride = {256, 16, 16, 1} , .addr = v12.addr + (((bx * 8)) * 16+((kk * 8)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((bx * 8)) * 16+((kk * 8)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info x_block_1 = {.shape = {1, 8, 1, 8}, .stride = x_block.stride, .addr = x_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = x_block.default_stride};
          tpu_gdma_cpy_S2L(x_block_1.addr, x.addr, &x_block_1.shape, (x_block_1.default_stride ? NULL : &x_block_1.stride), (x.default_stride ? NULL : &x.stride), DT_BFP16);
          __ppl_tensor_info gate_weight = {.shape = {1, 8, 1, 8}, .stride = {256, 16, 16, 1} , .addr = v14.addr + (((bz * 8)) * 16+((kk * 8)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((bz * 8)) * 16+((kk * 8)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info gate_weight_block_1 = {.shape = {1, 8, 1, 8}, .stride = gate_weight_block.stride, .addr = gate_weight_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gate_weight_block.default_stride};
          tpu_gdma_cpy_S2L(gate_weight_block_1.addr, gate_weight.addr, &gate_weight_block_1.shape, (gate_weight_block_1.default_stride ? NULL : &gate_weight_block_1.stride), (gate_weight.default_stride ? NULL : &gate_weight.stride), DT_BFP16);
          __ppl_tensor_info up_weight = {.shape = {1, 8, 1, 8}, .stride = {256, 16, 16, 1} , .addr = v13.addr + (((bz * 8)) * 16+((kk * 8)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((bz * 8)) * 16+((kk * 8)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
          __ppl_tensor_info up_weight_block_1 = {.shape = {1, 8, 1, 8}, .stride = up_weight_block.stride, .addr = up_weight_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = up_weight_block.default_stride};
          tpu_gdma_cpy_S2L(up_weight_block_1.addr, up_weight.addr, &up_weight_block_1.shape, (up_weight_block_1.default_stride ? NULL : &up_weight_block_1.stride), (up_weight.default_stride ? NULL : &up_weight.stride), DT_BFP16);
          {
          scalar_t proj_part_fp32_scalar_f32 = {.f32 = 0.000000e+00f};
          tpu_bdc_set_C(proj_part_fp32.addr, proj_part_fp32_scalar_f32, &proj_part_fp32.shape, (proj_part_fp32.default_stride ? NULL : &proj_part_fp32.stride), DT_FP32);
          }
          tpu_bdc_fp_mm_R_trans(proj_part_fp32.addr, x_block.addr, gate_weight_block.addr, 8, 8, 8, DT_FP32, DT_BFP16);
          tpu_bdc_fp_add( gate_out_block_fp32.addr, gate_out_block_fp32.addr, proj_part_fp32.addr, &gate_out_block_fp32.shape, (gate_out_block_fp32.default_stride ? NULL : &gate_out_block_fp32.stride), (gate_out_block_fp32.default_stride ? NULL : &gate_out_block_fp32.stride), (proj_part_fp32.default_stride ? NULL : &proj_part_fp32.stride), DT_FP32);
          {
          scalar_t proj_part_fp32_scalar_f32 = {.f32 = 0.000000e+00f};
          tpu_bdc_set_C(proj_part_fp32.addr, proj_part_fp32_scalar_f32, &proj_part_fp32.shape, (proj_part_fp32.default_stride ? NULL : &proj_part_fp32.stride), DT_FP32);
          }
          tpu_bdc_fp_mm_R_trans(proj_part_fp32.addr, x_block.addr, up_weight_block.addr, 8, 8, 8, DT_FP32, DT_BFP16);
          tpu_bdc_fp_add( up_out_block_fp32.addr, up_out_block_fp32.addr, proj_part_fp32.addr, &up_out_block_fp32.shape, (up_out_block_fp32.default_stride ? NULL : &up_out_block_fp32.stride), (up_out_block_fp32.default_stride ? NULL : &up_out_block_fp32.stride), (proj_part_fp32.default_stride ? NULL : &proj_part_fp32.stride), DT_FP32);
        }
        tpu_bdc_fp_mul_C( x_neg.addr, gate_out_block_fp32.addr, (scalar_t){.f32 = -1}, &x_neg.shape, (x_neg.default_stride ? NULL : &x_neg.stride), (gate_out_block_fp32.default_stride ? NULL : &gate_out_block_fp32.stride), DT_FP32);
        tpu_bdc_load_fp32_exp_coeff(exp_coeff.addr);
        tpu_bdc_load_fp32_exp_table(exp_table.addr);
        tpu_bdc_fp32_exp(x_neg.addr, x_neg.addr, exp_work0.addr, exp_work1.addr, exp_coeff.addr, exp_table.addr, &x_neg.shape);
        tpu_bdc_fp_add( x_neg_exp_1.addr, x_neg.addr, ones.addr, &x_neg_exp_1.shape, (x_neg_exp_1.default_stride ? NULL : &x_neg_exp_1.stride), (x_neg.default_stride ? NULL : &x_neg.stride), (ones.default_stride ? NULL : &ones.stride), DT_FP32);
        tpu_bdc_fp_div( silu_out_block.addr, gate_out_block_fp32.addr, x_neg_exp_1.addr, &silu_out_block.shape, (silu_out_block.default_stride ? NULL : &silu_out_block.stride), (gate_out_block_fp32.default_stride ? NULL : &gate_out_block_fp32.stride), (x_neg_exp_1.default_stride ? NULL : &x_neg_exp_1.stride), DT_FP32);
        tpu_bdc_fp_mul( gated_up_fp32.addr, silu_out_block.addr, up_out_block_fp32.addr, &gated_up_fp32.shape, (gated_up_fp32.default_stride ? NULL : &gated_up_fp32.stride), (silu_out_block.default_stride ? NULL : &silu_out_block.stride), (up_out_block_fp32.default_stride ? NULL : &up_out_block_fp32.stride), DT_FP32);
        __ppl_tensor_info gated_up_fp32_1 = {.shape = {1, 8, 1, 8}, .stride = gated_up_fp32.stride, .addr = gated_up_fp32.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gated_up_fp32.default_stride};
        __ppl_tensor_info gated_up_block_1 = {.shape = {1, 8, 1, 8}, .stride = gated_up_block.stride, .addr = gated_up_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = gated_up_block.default_stride};
        tpu_bdc_cast(gated_up_block_1.addr, gated_up_fp32_1.addr, &gated_up_block_1.shape, (gated_up_block_1.default_stride ? NULL : &gated_up_block_1.stride), (gated_up_fp32_1.default_stride ? NULL : &gated_up_fp32_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
        __ppl_tensor_info down_weight = {.shape = {1, 8, 1, 8}, .stride = {256, 16, 16, 1} , .addr = v15.addr + (((by * 8)) * 16+((bz * 8)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((by * 8)) * 16+((bz * 8)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info down_weight_block_1 = {.shape = {1, 8, 1, 8}, .stride = down_weight_block.stride, .addr = down_weight_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = down_weight_block.default_stride};
        tpu_gdma_cpy_S2L(down_weight_block_1.addr, down_weight.addr, &down_weight_block_1.shape, (down_weight_block_1.default_stride ? NULL : &down_weight_block_1.stride), (down_weight.default_stride ? NULL : &down_weight.stride), DT_BFP16);
        {
        scalar_t down_out_part_fp32_scalar_f32 = {.f32 = 0.000000e+00f};
        tpu_bdc_set_C(down_out_part_fp32.addr, down_out_part_fp32_scalar_f32, &down_out_part_fp32.shape, (down_out_part_fp32.default_stride ? NULL : &down_out_part_fp32.stride), DT_FP32);
        }
        tpu_bdc_fp_mm_R_trans(down_out_part_fp32.addr, gated_up_block.addr, down_weight_block.addr, 8, 8, 8, DT_FP32, DT_BFP16);
        tpu_bdc_fp_add( down_out_acc.addr, down_out_acc.addr, down_out_part_fp32.addr, &down_out_acc.shape, (down_out_acc.default_stride ? NULL : &down_out_acc.stride), (down_out_acc.default_stride ? NULL : &down_out_acc.stride), (down_out_part_fp32.default_stride ? NULL : &down_out_part_fp32.stride), DT_FP32);
      }
      __ppl_tensor_info down_out_acc_1 = {.shape = {1, 8, 1, 8}, .stride = down_out_acc.stride, .addr = down_out_acc.addr, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = down_out_acc.default_stride};
      __ppl_tensor_info down_out_block_1 = {.shape = {1, 8, 1, 8}, .stride = down_out_block.stride, .addr = down_out_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = down_out_block.default_stride};
      tpu_bdc_cast(down_out_block_1.addr, down_out_acc_1.addr, &down_out_block_1.shape, (down_out_block_1.default_stride ? NULL : &down_out_block_1.stride), (down_out_acc_1.default_stride ? NULL : &down_out_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info down_out_block_2 = {.shape = {1, 8, 1, 8}, .stride = down_out_block.stride, .addr = down_out_block.addr, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = down_out_block.default_stride};
      __ppl_tensor_info output = {.shape = {1, 8, 1, 8}, .stride = {256, 16, 16, 1} , .addr = v21.addr + (((bx * 8)) * 16+((by * 8)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((bx * 8)) * 16+((by * 8)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(output.addr, down_out_block_2.addr, &output.shape, (output.default_stride ? NULL : &output.stride), (down_out_block_2.default_stride ? NULL : &down_out_block_2.stride), DT_BFP16);
    }
  }
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
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  llama_mlp_forward(api->v1,
    api->v2,
    api->v3,
    api->v4,
    api->v5,
    api->v6,
    api->v7,
    api->v8,
    api->v9,
    api->v10,
    api->v11);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
