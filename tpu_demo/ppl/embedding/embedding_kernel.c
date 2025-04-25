#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

void main(global_addr_t ptr_output_v1, global_addr_t ptr_param_v2, global_addr_t ptr_index_v3) {
  __ppl_tensor_info v6 = {.shape = {1 ,7, 1, 1}, .stride = NULL, .addr = ptr_index_v3, .dtype = , .mode = 2, .align_mode = 0, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1 ,65, 1, 17}, .stride = NULL, .addr = ptr_param_v2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 2210, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v4 = {.shape = {1 ,7, 1, 17}, .stride = NULL, .addr = ptr_output_v1, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = 238, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info params_shared = {.shape = { 1, 65, 1, 17}, .stride = NULL, .addr = 16384, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 34, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info indices_shared = {.shape = { 1, 7, 1, 1}, .stride = NULL, .addr = 65536, .dtype = , .mode = 2, .align_mode = 1, .size = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info output_shared = {.shape = { 1, 7, 1, 17}, .stride = NULL, .addr = 49152, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info params_tmp_buffer = {.shape = { 1, 17, 1, 65}, .stride = NULL, .addr = 0, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 34, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info output_tmp_buffer = {.shape = { 1, 17, 1, 7}, .stride = NULL, .addr = 32768, .dtype = DT_FP16, .mode = 2, .align_mode = 1, .size = 2, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info params = {.shape = {1, 65, 1, 17}, .stride = {1105, 17, 17, 1} , .addr = v5.addr + ((0) * 17+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 17+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info params_shared_1 = {.shape = {1, 65, 1, 17}, .stride = NULL, .addr = params_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_gdma_cpy_S2L(params_shared_1.addr, params.addr, &params_shared_1.shape, (params_shared_1.default_stride ? NULL : &params_shared_1.stride), (params.default_stride ? NULL : &params.stride), DT_FP16);
  __ppl_tensor_info indices = {.shape = {1, 7, 1, 1}, .stride = {7, 1, 1, 1} , .addr = v6.addr + ((0) * 1+(0) * 1 ) * 2, .dtype = DT_UINT16, .mode = 2, .size = 1, .offset = ((0) * 1+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info indices_shared_1 = {.shape = {1, 7, 1, 1}, .stride = NULL, .addr = indices_shared.addr, .dtype = DT_UINT16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  tpu_gdma_cpy_S2L(indices_shared_1.addr, indices.addr, &indices_shared_1.shape, (indices_shared_1.default_stride ? NULL : &indices_shared_1.stride), (indices.default_stride ? NULL : &indices.stride), DT_UINT16);
  scalar_t output_shared_scalar_f16 = {.f16 = 0};
  tpu_bdc_set_C(output_shared.addr, output_shared_scalar_f16, &output_shared.shape, (output_shared.default_stride ? NULL : &output_shared.stride), DT_FP16);
  {
    int core_idx = tpu_core_index();
    int core_num = tpu_core_num();
    dim4 ori_output_shape = {1, 7, 1, 17};
    dim4 ori_params_shape = {1, 65, 1, 17};
    dim4 ori_index_shape = {1, 7, 1, 1};
    dim4 ori_output_stride = {119, 17,17, 1};
    dim4 ori_params_stride = {1105, 17,17, 1};
    dim4 ori_index_stride = {7, 1, 1, 1};
    dim4 output_shape = {1, 17, 1, 7};
    dim4 params_shape = {1, 17, 1, 65};
    dim4 index_shape = {1, 7, 1, 1};
    dim4 output_stride = {119, 7,7, 1};
    dim4 params_stride = {1105, 65,65, 1};
    dim4 index_stride = {7, 1, 1, 1};
    tpu_gdma_cpy_cw_trans_L2L(params_tmp_buffer.addr, params_shared.addr, &params_shape, &params_stride, &ori_params_stride, DT_UINT16);
    int inner_slice = (17 + core_num - 1) / core_num;
    int real_inner_slice = MIN(inner_slice, 17 - core_idx * inner_slice);
    if (inner_slice > 0) {
      dim4 params_subview_shape = {1, real_inner_slice, 1, 65};
      dim4 params_subview_stride = {real_inner_slice * 65, 65, 65, 1};
      dim4 output_subview_shape = {1, real_inner_slice, 1, 7};
      dim4 output_subview_stride = {7* real_inner_slice, 7, 7, 1};
      __ppl_tensor_info params_subview_shared = {.shape = params_subview_shape, .stride = params_subview_stride, .addr = params_shared.addr + core_idx * inner_slice * 2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = real_inner_slice * 65, .unsigned_flag = 0, .default_stride = true};
      __ppl_tensor_info output_subview_shared = {.shape = output_subview_shape, .stride = output_subview_stride, .addr = output_tmp_buffer.addr + core_idx * inner_slice * 2, .dtype = DT_FP16, .mode = 2, .align_mode = 0, .size = real_inner_slice * 7, .unsigned_flag = 0, .default_stride = true};
      tpu_bdc_w_gather(output_subview_shared.addr, params_tmp_buffer.addr, indices_shared.addr, &output_subview_shared.shape, params_shape.w, DT_FP16, DT_UINT16);
    };
    tpu_gdma_cpy_cw_trans_L2L(output_shared.addr, output_tmp_buffer.addr, &ori_output_shape, &ori_output_stride, &output_stride, DT_FP16);
  }  //section embedding
  __ppl_tensor_info output_shared_1 = {.shape = {1, 7, 1, 17}, .stride = NULL, .addr = output_shared.addr, .dtype = DT_FP16, .mode = 0, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info output = {.shape = {1, 7, 1, 17}, .stride = {119, 17, 17, 1} , .addr = v4.addr + ((0) * 17+(0) * 1 ) * 2, .dtype = DT_FP16, .mode = 2, .size = 1, .offset = ((0) * 17+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  tpu_gdma_cpy_L2S(output.addr, output_shared_1.addr, &output.shape, (output.default_stride ? NULL : &output.stride), (output_shared_1.default_stride ? NULL : &output_shared_1.stride), DT_FP16);
}

typedef struct {
  global_addr_t ptr_output_v1;
  global_addr_t ptr_param_v2;
  global_addr_t ptr_index_v3;
} tpu_kernel_api_main_args_t;
void main_kernel(const void * args) {
  tpu_kernel_api_main_args_t *api = (tpu_kernel_api_main_args_t*)args;
  main(api->ptr_output_v1,
    api->ptr_param_v2,
    api->ptr_index_v3);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
