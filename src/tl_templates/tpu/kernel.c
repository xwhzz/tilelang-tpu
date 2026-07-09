#include "ppl_helper.h"
#ifndef __PPL_HELPER__
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}
#endif  // __PPL_HELPER__

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

static inline int __ppl_div_up_i32(int x, int y) { return (x + y - 1) / y; }
static inline int __ppl_align_up_i32(int x, int y) { return __ppl_div_up_i32(x, y) * y; }
static inline int __ppl_dtype_bytes_for_stride(data_type_t dtype) {
  if (dtype == DT_FP32 || dtype == DT_TF32 || dtype == DT_INT32 || dtype == DT_UINT32) return 4;
  if (dtype == DT_FP16 || dtype == DT_BFP16 || dtype == DT_INT16 || dtype == DT_UINT16) return 2;
  return 1;
}
static inline void __ppl_static_aligned_stride(dim4 *stride, const dim4 *shape, data_type_t dtype) {
  int dtype_bytes = __ppl_dtype_bytes_for_stride(dtype);
  int align_elems = 64 / dtype_bytes;
  int hw_stride = shape->w;
  int c_stride = __ppl_align_up_i32(shape->h * shape->w, align_elems);
  stride->w = 1;
  stride->h = hw_stride;
  stride->c = c_stride;
  stride->n = __ppl_div_up_i32(shape->c, 64) * c_stride;
}

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4, global_addr_t v5, global_addr_t v6, global_addr_t v7, global_addr_t v8, global_addr_t v9, global_addr_t v10, global_addr_t v11, global_addr_t v12, global_addr_t v13, global_addr_t v14, global_addr_t v15, global_addr_t v16, global_addr_t v17) {
  __ppl_tensor_info v34 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v17, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v32 = {.shape = {1, 8, 1, 128} , .stride = {0}, .addr = v15, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 2048, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v31 = {.shape = {1, 8, 1, 192} , .stride = {0}, .addr = v14, .dtype = DT_FP32, .mode = 2, .align_mode = 0, .size = 6144, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v18 = {.shape = {1, 8, 1, 1536} , .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v19 = {.shape = {1, 192, 1, 512} , .stride = {0}, .addr = v2, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 196608, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v24 = {.shape = {1, 256, 1, 512} , .stride = {0}, .addr = v7, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 262144, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v21 = {.shape = {1, 192, 1, 1536} , .stride = {0}, .addr = v4, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 294912, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v33 = {.shape = {1, 192, 1, 512} , .stride = {0}, .addr = v16, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 196608, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v22 = {.shape = {1, 256, 1, 512} , .stride = {0}, .addr = v5, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 131072, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v20 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v23 = {.shape = {1, 192, 1, 1536} , .stride = {0}, .addr = v6, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 589824, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v27 = {.shape = {1, 8, 1, 64} , .stride = {0}, .addr = v10, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 1024, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v28 = {.shape = {1, 8, 1, 64} , .stride = {0}, .addr = v11, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 1024, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v25 = {.shape = {1, 192, 1, 512} , .stride = {0}, .addr = v8, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 196608, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v30 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v13, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v26 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v9, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v29 = {.shape = {1, 192, 1, 64} , .stride = {0}, .addr = v12, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 24576, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info q_nope_acc = {.shape = { 1, 8, 1, 128}, .stride = {0}, .addr = 256, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_nope_acc.stride, 0, &q_nope_acc.shape, DT_FP32);
  __ppl_tensor_info q_rope_src_acc = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src_acc.stride, 0, &q_rope_src_acc.shape, DT_FP32);
  __ppl_tensor_info q_nope = {.shape = { 1, 8, 1, 128}, .stride = {0}, .addr = 35072, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_nope.stride, 0, &q_nope.shape, DT_BFP16);
  __ppl_tensor_info q_rope_src = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_src.stride, 0, &q_rope_src.shape, DT_BFP16);
  __ppl_tensor_info q_rope = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 256, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope.stride, 0, &q_rope.shape, DT_BFP16);
  __ppl_tensor_info cos_q_shared = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 1024, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&cos_q_shared.stride, 0, &cos_q_shared.shape, DT_BFP16);
  __ppl_tensor_info sin_q_shared = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&sin_q_shared.stride, 0, &sin_q_shared.shape, DT_BFP16);
  __ppl_tensor_info q_rope_cos = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_cos.stride, 0, &q_rope_cos.shape, DT_BFP16);
  __ppl_tensor_info q_rope_sin = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_sin.stride, 0, &q_rope_sin.shape, DT_BFP16);
  __ppl_tensor_info q_rope_neg = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 32896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_neg.stride, 0, &q_rope_neg.shape, DT_BFP16);
  __ppl_tensor_info q_rope_neg_sin = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&q_rope_neg_sin.stride, 0, &q_rope_neg_sin.shape, DT_BFP16);
  __ppl_tensor_info ctx_acc = {.shape = { 1, 8, 1, 512}, .stride = {0}, .addr = 32896, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 2048, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&ctx_acc.stride, 0, &ctx_acc.shape, DT_FP32);
  __ppl_tensor_info ctx_cast = {.shape = { 1, 8, 1, 512}, .stride = {0}, .addr = 16576, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&ctx_cast.stride, 0, &ctx_cast.shape, DT_BFP16);
  __ppl_tensor_info scores_max = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 832, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&scores_max.stride, 0, &scores_max.shape, DT_FP32);
  __ppl_tensor_info logsum = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
  tpu_aligned_stride(&logsum.stride, 0, &logsum.shape, DT_FP32);
  for (int n = 0; n < 6; ++n) {
    for (int d = 0; d < 16; ++d) {
      __ppl_tensor_info cache_kv_tile = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&cache_kv_tile.stride, 0, &cache_kv_tile.shape, DT_BFP16);
      __ppl_tensor_info KV = {.shape = {1, 32, 1, 32} , .stride = {98304, 512, 512, 1} , .addr = v19.addr + (((n * 32)) * 512+((d * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n * 32)) * 512+((d * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info cache_kv_tile_1 = {.shape = {1, 32, 1, 32} , .stride = cache_kv_tile.stride, .addr = cache_kv_tile.addr + ((0) * cache_kv_tile.stride.c+(0) * cache_kv_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cache_kv_tile.stride.c+(0) * cache_kv_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cache_kv_tile.default_stride};
      tpu_gdma_cpy_S2L(cache_kv_tile_1.addr, KV.addr, &cache_kv_tile_1.shape, (cache_kv_tile_1.default_stride ? NULL : &cache_kv_tile_1.stride), (KV.default_stride ? NULL : &KV.stride), DT_BFP16);
      __ppl_tensor_info cache_kv_tile_2 = {.shape = {1, 32, 1, 32} , .stride = cache_kv_tile.stride, .addr = cache_kv_tile.addr + ((0) * cache_kv_tile.stride.c+(0) * cache_kv_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cache_kv_tile.stride.c+(0) * cache_kv_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cache_kv_tile.default_stride};
      __ppl_tensor_info KVcache_OUT = {.shape = {1, 32, 1, 32} , .stride = {98304, 512, 512, 1} , .addr = v33.addr + (((n * 32)) * 512+((d * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n * 32)) * 512+((d * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      tpu_gdma_cpy_L2S(KVcache_OUT.addr, cache_kv_tile_2.addr, &KVcache_OUT.shape, (KVcache_OUT.default_stride ? NULL : &KVcache_OUT.stride), (cache_kv_tile_2.default_stride ? NULL : &cache_kv_tile_2.stride), DT_BFP16);
    }
  }
  __ppl_tensor_info COS_Q = {.shape = {1, 8, 1, 64} , .stride = {512, 64, 64, 1} , .addr = v27.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info cos_q_shared_1 = {.shape = {1, 8, 1, 64} , .stride = cos_q_shared.stride, .addr = cos_q_shared.addr + ((0) * cos_q_shared.stride.c+(0) * cos_q_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_q_shared.stride.c+(0) * cos_q_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_q_shared.default_stride};
  tpu_gdma_cpy_S2L(cos_q_shared_1.addr, COS_Q.addr, &cos_q_shared_1.shape, (cos_q_shared_1.default_stride ? NULL : &cos_q_shared_1.stride), (COS_Q.default_stride ? NULL : &COS_Q.stride), DT_BFP16);
  __ppl_tensor_info SIN_Q = {.shape = {1, 8, 1, 64} , .stride = {512, 64, 64, 1} , .addr = v28.addr + ((0) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
  __ppl_tensor_info sin_q_shared_1 = {.shape = {1, 8, 1, 64} , .stride = sin_q_shared.stride, .addr = sin_q_shared.addr + ((0) * sin_q_shared.stride.c+(0) * sin_q_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_q_shared.stride.c+(0) * sin_q_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_q_shared.default_stride};
  tpu_gdma_cpy_S2L(sin_q_shared_1.addr, SIN_Q.addr, &sin_q_shared_1.shape, (sin_q_shared_1.default_stride ? NULL : &sin_q_shared_1.stride), (SIN_Q.default_stride ? NULL : &SIN_Q.stride), DT_BFP16);
  {
  scalar_t q_nope_acc_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(q_nope_acc.addr, q_nope_acc_scalar_f32, &q_nope_acc.shape, (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), DT_FP32);
  }
  {
  scalar_t q_rope_src_acc_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(q_rope_src_acc.addr, q_rope_src_acc_scalar_f32, &q_rope_src_acc.shape, (q_rope_src_acc.default_stride ? NULL : &q_rope_src_acc.stride), DT_FP32);
  }
  for (int k = 0; k < 48; ++k) {
    __ppl_tensor_info q_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_tile.stride, 0, &q_tile.shape, DT_BFP16);
    __ppl_tensor_info wuq_nope_fp8 = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 1152, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_nope_fp8.stride, 0, &wuq_nope_fp8.shape, DT_FP8E4M3);
    __ppl_tensor_info wuq_nope_scale = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 16448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_nope_scale.stride, 0, &wuq_nope_scale.shape, DT_BFP16);
    __ppl_tensor_info wuq_nope_tile = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_nope_tile.stride, 0, &wuq_nope_tile.shape, DT_BFP16);
    __ppl_tensor_info wuq_rope_fp8 = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 192, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_rope_fp8.stride, 0, &wuq_rope_fp8.shape, DT_FP8E4M3);
    __ppl_tensor_info wuq_rope_scale = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 16448, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_rope_scale.stride, 0, &wuq_rope_scale.shape, DT_BFP16);
    __ppl_tensor_info wuq_rope_tile = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&wuq_rope_tile.stride, 0, &wuq_rope_tile.shape, DT_BFP16);
    __ppl_tensor_info q_nope_part = {.shape = { 1, 8, 1, 128}, .stride = {0}, .addr = 16448, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_nope_part.stride, 0, &q_nope_part.shape, DT_FP32);
    __ppl_tensor_info q_rope_part = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 16448, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&q_rope_part.stride, 0, &q_rope_part.shape, DT_FP32);
    __ppl_tensor_info Q = {.shape = {1, 8, 1, 32} , .stride = {12288, 1536, 1536, 1} , .addr = v18.addr + ((0) * 1536+((k * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info q_tile_1 = {.shape = {1, 8, 1, 32} , .stride = q_tile.stride, .addr = q_tile.addr + ((0) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_tile.stride.c+(0) * q_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_tile.default_stride};
    tpu_gdma_cpy_S2L(q_tile_1.addr, Q.addr, &q_tile_1.shape, (q_tile_1.default_stride ? NULL : &q_tile_1.stride), (Q.default_stride ? NULL : &Q.stride), DT_BFP16);
    __ppl_tensor_info WUQ = {.shape = {1, 128, 1, 32} , .stride = {294912, 1536, 1536, 1} , .addr = v21.addr + ((0) * 1536+((k * 32)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((0) * 1536+((k * 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wuq_nope_fp8_1 = {.shape = {1, 128, 1, 32} , .stride = wuq_nope_fp8.stride, .addr = wuq_nope_fp8.addr + ((0) * wuq_nope_fp8.stride.c+(0) * wuq_nope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_nope_fp8.stride.c+(0) * wuq_nope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_nope_fp8.default_stride};
    tpu_gdma_cpy_S2L(wuq_nope_fp8_1.addr, WUQ.addr, &wuq_nope_fp8_1.shape, (wuq_nope_fp8_1.default_stride ? NULL : &wuq_nope_fp8_1.stride), (WUQ.default_stride ? NULL : &WUQ.stride), DT_FP8E4M3);
    __ppl_tensor_info wuq_nope_fp8_2 = {.shape = {1, 128, 1, 32} , .stride = wuq_nope_fp8.stride, .addr = wuq_nope_fp8.addr + ((0) * wuq_nope_fp8.stride.c+(0) * wuq_nope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_nope_fp8.stride.c+(0) * wuq_nope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_nope_fp8.default_stride};
    __ppl_tensor_info wuq_nope_tile_1 = {.shape = {1, 128, 1, 32} , .stride = wuq_nope_tile.stride, .addr = wuq_nope_tile.addr + ((0) * wuq_nope_tile.stride.c+(0) * wuq_nope_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_nope_tile.stride.c+(0) * wuq_nope_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_nope_tile.default_stride};
    tpu_bdc_cast(wuq_nope_tile_1.addr, wuq_nope_fp8_2.addr, &wuq_nope_tile_1.shape, (wuq_nope_tile_1.default_stride ? NULL : &wuq_nope_tile_1.stride), (wuq_nope_fp8_2.default_stride ? NULL : &wuq_nope_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
    __ppl_tensor_info WUQ_SCALE = {.shape = {1, 128, 1, 32} , .stride = {294912, 1536, 1536, 1} , .addr = v23.addr + ((0) * 1536+((k * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((k * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wuq_nope_scale_1 = {.shape = {1, 128, 1, 32} , .stride = wuq_nope_scale.stride, .addr = wuq_nope_scale.addr + ((0) * wuq_nope_scale.stride.c+(0) * wuq_nope_scale.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_nope_scale.stride.c+(0) * wuq_nope_scale.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_nope_scale.default_stride};
    tpu_gdma_cpy_S2L(wuq_nope_scale_1.addr, WUQ_SCALE.addr, &wuq_nope_scale_1.shape, (wuq_nope_scale_1.default_stride ? NULL : &wuq_nope_scale_1.stride), (WUQ_SCALE.default_stride ? NULL : &WUQ_SCALE.stride), DT_BFP16);
    tpu_bdc_fp_mul( wuq_nope_tile.addr, wuq_nope_tile.addr, wuq_nope_scale.addr, &wuq_nope_tile.shape, (wuq_nope_tile.default_stride ? NULL : &wuq_nope_tile.stride), (wuq_nope_tile.default_stride ? NULL : &wuq_nope_tile.stride), (wuq_nope_scale.default_stride ? NULL : &wuq_nope_scale.stride), DT_BFP16);
    {
    scalar_t q_nope_part_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_nope_part.addr, q_nope_part_scalar_f32, &q_nope_part.shape, (q_nope_part.default_stride ? NULL : &q_nope_part.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(q_nope_part.addr, q_tile.addr, wuq_nope_tile.addr, 8, 32, 128, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( q_nope_acc.addr, q_nope_acc.addr, q_nope_part.addr, &q_nope_acc.shape, (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), (q_nope_acc.default_stride ? NULL : &q_nope_acc.stride), (q_nope_part.default_stride ? NULL : &q_nope_part.stride), DT_FP32);
    __ppl_tensor_info WUQ_1 = {.shape = {1, 64, 1, 32} , .stride = {294912, 1536, 1536, 1} , .addr = v21.addr + ((128) * 1536+((k * 32)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((128) * 1536+((k * 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wuq_rope_fp8_1 = {.shape = {1, 64, 1, 32} , .stride = wuq_rope_fp8.stride, .addr = wuq_rope_fp8.addr + ((0) * wuq_rope_fp8.stride.c+(0) * wuq_rope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_rope_fp8.stride.c+(0) * wuq_rope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_rope_fp8.default_stride};
    tpu_gdma_cpy_S2L(wuq_rope_fp8_1.addr, WUQ_1.addr, &wuq_rope_fp8_1.shape, (wuq_rope_fp8_1.default_stride ? NULL : &wuq_rope_fp8_1.stride), (WUQ_1.default_stride ? NULL : &WUQ_1.stride), DT_FP8E4M3);
    __ppl_tensor_info wuq_rope_fp8_2 = {.shape = {1, 64, 1, 32} , .stride = wuq_rope_fp8.stride, .addr = wuq_rope_fp8.addr + ((0) * wuq_rope_fp8.stride.c+(0) * wuq_rope_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wuq_rope_fp8.stride.c+(0) * wuq_rope_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wuq_rope_fp8.default_stride};
    __ppl_tensor_info wuq_rope_tile_1 = {.shape = {1, 64, 1, 32} , .stride = wuq_rope_tile.stride, .addr = wuq_rope_tile.addr + ((0) * wuq_rope_tile.stride.c+(0) * wuq_rope_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_rope_tile.stride.c+(0) * wuq_rope_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_rope_tile.default_stride};
    tpu_bdc_cast(wuq_rope_tile_1.addr, wuq_rope_fp8_2.addr, &wuq_rope_tile_1.shape, (wuq_rope_tile_1.default_stride ? NULL : &wuq_rope_tile_1.stride), (wuq_rope_fp8_2.default_stride ? NULL : &wuq_rope_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
    __ppl_tensor_info WUQ_SCALE_1 = {.shape = {1, 64, 1, 32} , .stride = {294912, 1536, 1536, 1} , .addr = v23.addr + ((128) * 1536+((k * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((128) * 1536+((k * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info wuq_rope_scale_1 = {.shape = {1, 64, 1, 32} , .stride = wuq_rope_scale.stride, .addr = wuq_rope_scale.addr + ((0) * wuq_rope_scale.stride.c+(0) * wuq_rope_scale.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wuq_rope_scale.stride.c+(0) * wuq_rope_scale.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wuq_rope_scale.default_stride};
    tpu_gdma_cpy_S2L(wuq_rope_scale_1.addr, WUQ_SCALE_1.addr, &wuq_rope_scale_1.shape, (wuq_rope_scale_1.default_stride ? NULL : &wuq_rope_scale_1.stride), (WUQ_SCALE_1.default_stride ? NULL : &WUQ_SCALE_1.stride), DT_BFP16);
    tpu_bdc_fp_mul( wuq_rope_tile.addr, wuq_rope_tile.addr, wuq_rope_scale.addr, &wuq_rope_tile.shape, (wuq_rope_tile.default_stride ? NULL : &wuq_rope_tile.stride), (wuq_rope_tile.default_stride ? NULL : &wuq_rope_tile.stride), (wuq_rope_scale.default_stride ? NULL : &wuq_rope_scale.stride), DT_BFP16);
    {
    scalar_t q_rope_part_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(q_rope_part.addr, q_rope_part_scalar_f32, &q_rope_part.shape, (q_rope_part.default_stride ? NULL : &q_rope_part.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(q_rope_part.addr, q_tile.addr, wuq_rope_tile.addr, 8, 32, 64, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( q_rope_src_acc.addr, q_rope_src_acc.addr, q_rope_part.addr, &q_rope_src_acc.shape, (q_rope_src_acc.default_stride ? NULL : &q_rope_src_acc.stride), (q_rope_src_acc.default_stride ? NULL : &q_rope_src_acc.stride), (q_rope_part.default_stride ? NULL : &q_rope_part.stride), DT_FP32);
  }
  __ppl_tensor_info q_nope_acc_1 = {.shape = {1, 8, 1, 128} , .stride = q_nope_acc.stride, .addr = q_nope_acc.addr + ((0) * q_nope_acc.stride.c+(0) * q_nope_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_nope_acc.stride.c+(0) * q_nope_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_nope_acc.default_stride};
  __ppl_tensor_info q_nope_1 = {.shape = {1, 8, 1, 128} , .stride = q_nope.stride, .addr = q_nope.addr + ((0) * q_nope.stride.c+(0) * q_nope.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_nope.stride.c+(0) * q_nope.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_nope.default_stride};
  tpu_bdc_cast(q_nope_1.addr, q_nope_acc_1.addr, &q_nope_1.shape, (q_nope_1.default_stride ? NULL : &q_nope_1.stride), (q_nope_acc_1.default_stride ? NULL : &q_nope_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
  __ppl_tensor_info q_rope_src_acc_1 = {.shape = {1, 8, 1, 64} , .stride = q_rope_src_acc.stride, .addr = q_rope_src_acc.addr + ((0) * q_rope_src_acc.stride.c+(0) * q_rope_src_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_rope_src_acc.stride.c+(0) * q_rope_src_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_rope_src_acc.default_stride};
  __ppl_tensor_info q_rope_src_1 = {.shape = {1, 8, 1, 64} , .stride = q_rope_src.stride, .addr = q_rope_src.addr + ((0) * q_rope_src.stride.c+(0) * q_rope_src.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_rope_src.stride.c+(0) * q_rope_src.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_rope_src.default_stride};
  tpu_bdc_cast(q_rope_src_1.addr, q_rope_src_acc_1.addr, &q_rope_src_1.shape, (q_rope_src_1.default_stride ? NULL : &q_rope_src_1.stride), (q_rope_src_acc_1.default_stride ? NULL : &q_rope_src_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
  tpu_bdc_fp_mul( q_rope_cos.addr, q_rope_src.addr, cos_q_shared.addr, &q_rope_cos.shape, (q_rope_cos.default_stride ? NULL : &q_rope_cos.stride), (q_rope_src.default_stride ? NULL : &q_rope_src.stride), (cos_q_shared.default_stride ? NULL : &cos_q_shared.stride), DT_BFP16);
  tpu_bdc_fp_mul( q_rope_sin.addr, q_rope_src.addr, sin_q_shared.addr, &q_rope_sin.shape, (q_rope_sin.default_stride ? NULL : &q_rope_sin.stride), (q_rope_src.default_stride ? NULL : &q_rope_src.stride), (sin_q_shared.default_stride ? NULL : &sin_q_shared.stride), DT_BFP16);
  {
  scalar_t q_rope_neg_scalar_DT_BFP16 = {.f32 = (float)-1};
  q_rope_neg_scalar_DT_BFP16 = tpu_cast(q_rope_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
  tpu_bdc_fp_mul_C( q_rope_neg.addr, q_rope_src.addr, q_rope_neg_scalar_DT_BFP16, &q_rope_neg.shape, (q_rope_neg.default_stride ? NULL : &q_rope_neg.stride), (q_rope_src.default_stride ? NULL : &q_rope_src.stride), DT_BFP16);
  }
  tpu_bdc_fp_mul( q_rope_neg_sin.addr, q_rope_neg.addr, sin_q_shared.addr, &q_rope_neg_sin.shape, (q_rope_neg_sin.default_stride ? NULL : &q_rope_neg_sin.stride), (q_rope_neg.default_stride ? NULL : &q_rope_neg.stride), (sin_q_shared.default_stride ? NULL : &sin_q_shared.stride), DT_BFP16);
  {
  dim4 half_stride;
  __ppl_static_aligned_stride(&half_stride, &q_rope.shape, DT_BFP16);
  half_stride.w *= 2;
  dim4 half_shape = {.n = q_rope.shape.n, .c = q_rope.shape.c, .h = q_rope.shape.h, .w = q_rope.shape.w};
  half_shape.w /= 2;
  tpu_bdc_fp_add( q_rope.addr, q_rope_cos.addr, q_rope_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
  tpu_bdc_fp_add( q_rope.addr + 2, q_rope_cos.addr + 2, q_rope_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
  }
  {
  scalar_t ctx_acc_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(ctx_acc.addr, ctx_acc_scalar_f32, &ctx_acc.shape, (ctx_acc.default_stride ? NULL : &ctx_acc.stride), DT_FP32);
  }
  {
  scalar_t logsum_scalar_f32 = {.f32 = 0.000000e+00f};
  tpu_bdc_set_C(logsum.addr, logsum_scalar_f32, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), DT_FP32);
  }
  {
  scalar_t scores_max_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
  tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
  }
  for (int n_1 = 0; n_1 < 6; ++n_1) {
    __ppl_tensor_info pe_raw_tile = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_raw_tile.stride, 0, &pe_raw_tile.shape, DT_BFP16);
    __ppl_tensor_info cos_all_tile = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 34944, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&cos_all_tile.stride, 0, &cos_all_tile.shape, DT_BFP16);
    __ppl_tensor_info sin_all_tile = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&sin_all_tile.stride, 0, &sin_all_tile.shape, DT_BFP16);
    __ppl_tensor_info pe_rope_cos = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_rope_cos.stride, 0, &pe_rope_cos.shape, DT_BFP16);
    __ppl_tensor_info pe_rope_sin = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 34944, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_rope_sin.stride, 0, &pe_rope_sin.shape, DT_BFP16);
    __ppl_tensor_info pe_rope_neg = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_rope_neg.stride, 0, &pe_rope_neg.shape, DT_BFP16);
    __ppl_tensor_info pe_rope_neg_sin = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 49152, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_rope_neg_sin.stride, 0, &pe_rope_neg_sin.shape, DT_BFP16);
    __ppl_tensor_info pe_rope_tile = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_rope_tile.stride, 0, &pe_rope_tile.shape, DT_BFP16);
    __ppl_tensor_info kv_value_tile = {.shape = { 1, 32, 1, 512}, .stride = {0}, .addr = 832, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 1024, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&kv_value_tile.stride, 0, &kv_value_tile.shape, DT_BFP16);
    __ppl_tensor_info pe_score_tile = {.shape = { 1, 32, 1, 64}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&pe_score_tile.stride, 0, &pe_score_tile.shape, DT_BFP16);
    __ppl_tensor_info score_latent_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_latent_tile.stride, 0, &score_latent_tile.shape, DT_FP32);
    __ppl_tensor_info score_rope_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 34944, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_rope_tile.stride, 0, &score_rope_tile.shape, DT_FP32);
    __ppl_tensor_info score_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile.stride, 0, &score_tile.shape, DT_FP32);
    __ppl_tensor_info score_tile_with_prev = {.shape = { 1, 8, 1, 64}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&score_tile_with_prev.stride, 0, &score_tile_with_prev.shape, DT_FP32);
    __ppl_tensor_info mask_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&mask_tile.stride, 0, &mask_tile.shape, DT_FP32);
    __ppl_tensor_info prob_tile = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prob_tile.stride, 0, &prob_tile.shape, DT_BFP16);
    __ppl_tensor_info scores_max_prev = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 256, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_max_prev.stride, 0, &scores_max_prev.shape, DT_FP32);
    __ppl_tensor_info scores_scale = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 16512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_scale.stride, 0, &scores_scale.shape, DT_FP32);
    __ppl_tensor_info scores_sum = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&scores_sum.stride, 0, &scores_sum.shape, DT_FP32);
    __ppl_tensor_info prev_scaled = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&prev_scaled.stride, 0, &prev_scaled.shape, DT_FP32);
    __ppl_tensor_info curr_scaled = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&curr_scaled.stride, 0, &curr_scaled.shape, DT_FP32);
    __ppl_tensor_info work_scale0 = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_scale0.stride, 0, &work_scale0.shape, DT_FP32);
    __ppl_tensor_info work_scale1 = {.shape = { 1, 8, 1, 1}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_scale1.stride, 0, &work_scale1.shape, DT_FP32);
    __ppl_tensor_info work_score0 = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 49152, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score0.stride, 0, &work_score0.shape, DT_FP32);
    __ppl_tensor_info work_score1 = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 65536, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&work_score1.stride, 0, &work_score1.shape, DT_FP32);
    __ppl_tensor_info coeff = {.shape = { 1, 64, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&coeff.stride, 0, &coeff.shape, DT_FP32);
    __ppl_tensor_info table = {.shape = { 1, 64, 1, 192}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 768, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&table.stride, 0, &table.shape, DT_FP32);
    __ppl_tensor_info PE_RAW = {.shape = {1, 32, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v20.addr + (((n_1 * 32)) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info pe_raw_tile_1 = {.shape = {1, 32, 1, 64} , .stride = pe_raw_tile.stride, .addr = pe_raw_tile.addr + ((0) * pe_raw_tile.stride.c+(0) * pe_raw_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_raw_tile.stride.c+(0) * pe_raw_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_raw_tile.default_stride};
    tpu_gdma_cpy_S2L(pe_raw_tile_1.addr, PE_RAW.addr, &pe_raw_tile_1.shape, (pe_raw_tile_1.default_stride ? NULL : &pe_raw_tile_1.stride), (PE_RAW.default_stride ? NULL : &PE_RAW.stride), DT_BFP16);
    __ppl_tensor_info COS_ALL = {.shape = {1, 32, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v29.addr + (((n_1 * 32)) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info cos_all_tile_1 = {.shape = {1, 32, 1, 64} , .stride = cos_all_tile.stride, .addr = cos_all_tile.addr + ((0) * cos_all_tile.stride.c+(0) * cos_all_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * cos_all_tile.stride.c+(0) * cos_all_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = cos_all_tile.default_stride};
    tpu_gdma_cpy_S2L(cos_all_tile_1.addr, COS_ALL.addr, &cos_all_tile_1.shape, (cos_all_tile_1.default_stride ? NULL : &cos_all_tile_1.stride), (COS_ALL.default_stride ? NULL : &COS_ALL.stride), DT_BFP16);
    __ppl_tensor_info SIN_ALL = {.shape = {1, 32, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v30.addr + (((n_1 * 32)) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info sin_all_tile_1 = {.shape = {1, 32, 1, 64} , .stride = sin_all_tile.stride, .addr = sin_all_tile.addr + ((0) * sin_all_tile.stride.c+(0) * sin_all_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * sin_all_tile.stride.c+(0) * sin_all_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = sin_all_tile.default_stride};
    tpu_gdma_cpy_S2L(sin_all_tile_1.addr, SIN_ALL.addr, &sin_all_tile_1.shape, (sin_all_tile_1.default_stride ? NULL : &sin_all_tile_1.stride), (SIN_ALL.default_stride ? NULL : &SIN_ALL.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_rope_cos.addr, pe_raw_tile.addr, cos_all_tile.addr, &pe_rope_cos.shape, (pe_rope_cos.default_stride ? NULL : &pe_rope_cos.stride), (pe_raw_tile.default_stride ? NULL : &pe_raw_tile.stride), (cos_all_tile.default_stride ? NULL : &cos_all_tile.stride), DT_BFP16);
    tpu_bdc_fp_mul( pe_rope_sin.addr, pe_raw_tile.addr, sin_all_tile.addr, &pe_rope_sin.shape, (pe_rope_sin.default_stride ? NULL : &pe_rope_sin.stride), (pe_raw_tile.default_stride ? NULL : &pe_raw_tile.stride), (sin_all_tile.default_stride ? NULL : &sin_all_tile.stride), DT_BFP16);
    {
    scalar_t pe_rope_neg_scalar_DT_BFP16 = {.f32 = (float)-1};
    pe_rope_neg_scalar_DT_BFP16 = tpu_cast(pe_rope_neg_scalar_DT_BFP16, DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    tpu_bdc_fp_mul_C( pe_rope_neg.addr, pe_raw_tile.addr, pe_rope_neg_scalar_DT_BFP16, &pe_rope_neg.shape, (pe_rope_neg.default_stride ? NULL : &pe_rope_neg.stride), (pe_raw_tile.default_stride ? NULL : &pe_raw_tile.stride), DT_BFP16);
    }
    tpu_bdc_fp_mul( pe_rope_neg_sin.addr, pe_rope_neg.addr, sin_all_tile.addr, &pe_rope_neg_sin.shape, (pe_rope_neg_sin.default_stride ? NULL : &pe_rope_neg_sin.stride), (pe_rope_neg.default_stride ? NULL : &pe_rope_neg.stride), (sin_all_tile.default_stride ? NULL : &sin_all_tile.stride), DT_BFP16);
    {
    dim4 half_stride;
    __ppl_static_aligned_stride(&half_stride, &pe_rope_tile.shape, DT_BFP16);
    half_stride.w *= 2;
    dim4 half_shape = {.n = pe_rope_tile.shape.n, .c = pe_rope_tile.shape.c, .h = pe_rope_tile.shape.h, .w = pe_rope_tile.shape.w};
    half_shape.w /= 2;
    tpu_bdc_fp_add( pe_rope_tile.addr, pe_rope_cos.addr, pe_rope_neg_sin.addr + 2, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    tpu_bdc_fp_add( pe_rope_tile.addr + 2, pe_rope_cos.addr + 2, pe_rope_sin.addr, &half_shape, &half_stride, &half_stride, &half_stride, DT_BFP16);
    }
    __ppl_tensor_info pe_rope_tile_1 = {.shape = {1, 32, 1, 64} , .stride = pe_rope_tile.stride, .addr = pe_rope_tile.addr + ((0) * pe_rope_tile.stride.c+(0) * pe_rope_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_rope_tile.stride.c+(0) * pe_rope_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_rope_tile.default_stride};
    __ppl_tensor_info PEcache_OUT = {.shape = {1, 32, 1, 64} , .stride = {12288, 64, 64, 1} , .addr = v34.addr + (((n_1 * 32)) * 64+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 64+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(PEcache_OUT.addr, pe_rope_tile_1.addr, &PEcache_OUT.shape, (PEcache_OUT.default_stride ? NULL : &PEcache_OUT.stride), (pe_rope_tile_1.default_stride ? NULL : &pe_rope_tile_1.stride), DT_BFP16);
    {
    scalar_t score_latent_tile_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_latent_tile.addr, score_latent_tile_scalar_f32, &score_latent_tile.shape, (score_latent_tile.default_stride ? NULL : &score_latent_tile.stride), DT_FP32);
    }
    for (int d_1 = 0; d_1 < 16; ++d_1) {
      __ppl_tensor_info wukv_k_fp8 = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 512, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_k_fp8.stride, 0, &wukv_k_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_k_scale = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_k_scale.stride, 0, &wukv_k_scale.shape, DT_BFP16);
      __ppl_tensor_info wukv_k_tile = {.shape = { 1, 128, 1, 32}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_k_tile.stride, 0, &wukv_k_tile.shape, DT_BFP16);
      __ppl_tensor_info q_wukv_part = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 512, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_wukv_part.stride, 0, &q_wukv_part.shape, DT_FP32);
      __ppl_tensor_info q_wukv_cast = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 128, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&q_wukv_cast.stride, 0, &q_wukv_cast.shape, DT_BFP16);
      __ppl_tensor_info kv_k_tile = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 34944, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&kv_k_tile.stride, 0, &kv_k_tile.shape, DT_BFP16);
      __ppl_tensor_info score_latent_part = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&score_latent_part.stride, 0, &score_latent_part.shape, DT_FP32);
      __ppl_tensor_info WUKV = {.shape = {1, 128, 1, 32} , .stride = {131072, 512, 512, 1} , .addr = v22.addr + ((0) * 512+((d_1 * 32)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((0) * 512+((d_1 * 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_k_fp8_1 = {.shape = {1, 128, 1, 32} , .stride = wukv_k_fp8.stride, .addr = wukv_k_fp8.addr + ((0) * wukv_k_fp8.stride.c+(0) * wukv_k_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_k_fp8.stride.c+(0) * wukv_k_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_k_fp8.default_stride};
      tpu_gdma_cpy_S2L(wukv_k_fp8_1.addr, WUKV.addr, &wukv_k_fp8_1.shape, (wukv_k_fp8_1.default_stride ? NULL : &wukv_k_fp8_1.stride), (WUKV.default_stride ? NULL : &WUKV.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_k_fp8_2 = {.shape = {1, 128, 1, 32} , .stride = wukv_k_fp8.stride, .addr = wukv_k_fp8.addr + ((0) * wukv_k_fp8.stride.c+(0) * wukv_k_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_k_fp8.stride.c+(0) * wukv_k_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_k_fp8.default_stride};
      __ppl_tensor_info wukv_k_tile_1 = {.shape = {1, 128, 1, 32} , .stride = wukv_k_tile.stride, .addr = wukv_k_tile.addr + ((0) * wukv_k_tile.stride.c+(0) * wukv_k_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_k_tile.stride.c+(0) * wukv_k_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_k_tile.default_stride};
      tpu_bdc_cast(wukv_k_tile_1.addr, wukv_k_fp8_2.addr, &wukv_k_tile_1.shape, (wukv_k_tile_1.default_stride ? NULL : &wukv_k_tile_1.stride), (wukv_k_fp8_2.default_stride ? NULL : &wukv_k_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info WUKV_SCALE = {.shape = {1, 128, 1, 32} , .stride = {131072, 512, 512, 1} , .addr = v24.addr + ((0) * 512+((d_1 * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 512+((d_1 * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_k_scale_1 = {.shape = {1, 128, 1, 32} , .stride = wukv_k_scale.stride, .addr = wukv_k_scale.addr + ((0) * wukv_k_scale.stride.c+(0) * wukv_k_scale.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_k_scale.stride.c+(0) * wukv_k_scale.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_k_scale.default_stride};
      tpu_gdma_cpy_S2L(wukv_k_scale_1.addr, WUKV_SCALE.addr, &wukv_k_scale_1.shape, (wukv_k_scale_1.default_stride ? NULL : &wukv_k_scale_1.stride), (WUKV_SCALE.default_stride ? NULL : &WUKV_SCALE.stride), DT_BFP16);
      tpu_bdc_fp_mul( wukv_k_tile.addr, wukv_k_tile.addr, wukv_k_scale.addr, &wukv_k_tile.shape, (wukv_k_tile.default_stride ? NULL : &wukv_k_tile.stride), (wukv_k_tile.default_stride ? NULL : &wukv_k_tile.stride), (wukv_k_scale.default_stride ? NULL : &wukv_k_scale.stride), DT_BFP16);
      {
      scalar_t q_wukv_part_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(q_wukv_part.addr, q_wukv_part_scalar_f32, &q_wukv_part.shape, (q_wukv_part.default_stride ? NULL : &q_wukv_part.stride), DT_FP32);
      }
      tpu_bdc_fp_mm(q_wukv_part.addr, q_nope.addr, wukv_k_tile.addr, 8, 128, 32, DT_FP32, DT_BFP16, true);
      __ppl_tensor_info q_wukv_part_1 = {.shape = {1, 8, 1, 32} , .stride = q_wukv_part.stride, .addr = q_wukv_part.addr + ((0) * q_wukv_part.stride.c+(0) * q_wukv_part.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * q_wukv_part.stride.c+(0) * q_wukv_part.stride.w ) * 4, .unsigned_flag = 0, .default_stride = q_wukv_part.default_stride};
      __ppl_tensor_info q_wukv_cast_1 = {.shape = {1, 8, 1, 32} , .stride = q_wukv_cast.stride, .addr = q_wukv_cast.addr + ((0) * q_wukv_cast.stride.c+(0) * q_wukv_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * q_wukv_cast.stride.c+(0) * q_wukv_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = q_wukv_cast.default_stride};
      tpu_bdc_cast(q_wukv_cast_1.addr, q_wukv_part_1.addr, &q_wukv_cast_1.shape, (q_wukv_cast_1.default_stride ? NULL : &q_wukv_cast_1.stride), (q_wukv_part_1.default_stride ? NULL : &q_wukv_part_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
      __ppl_tensor_info KVcache_OUT_1 = {.shape = {1, 32, 1, 32} , .stride = {98304, 512, 512, 1} , .addr = v33.addr + (((n_1 * 32)) * 512+((d_1 * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 512+((d_1 * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info kv_k_tile_1 = {.shape = {1, 32, 1, 32} , .stride = kv_k_tile.stride, .addr = kv_k_tile.addr + ((0) * kv_k_tile.stride.c+(0) * kv_k_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_k_tile.stride.c+(0) * kv_k_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_k_tile.default_stride};
      tpu_gdma_cpy_S2L(kv_k_tile_1.addr, KVcache_OUT_1.addr, &kv_k_tile_1.shape, (kv_k_tile_1.default_stride ? NULL : &kv_k_tile_1.stride), (KVcache_OUT_1.default_stride ? NULL : &KVcache_OUT_1.stride), DT_BFP16);
      {
      scalar_t score_latent_part_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(score_latent_part.addr, score_latent_part_scalar_f32, &score_latent_part.shape, (score_latent_part.default_stride ? NULL : &score_latent_part.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(score_latent_part.addr, q_wukv_cast.addr, kv_k_tile.addr, 8, 32, 32, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( score_latent_tile.addr, score_latent_tile.addr, score_latent_part.addr, &score_latent_tile.shape, (score_latent_tile.default_stride ? NULL : &score_latent_tile.stride), (score_latent_tile.default_stride ? NULL : &score_latent_tile.stride), (score_latent_part.default_stride ? NULL : &score_latent_part.stride), DT_FP32);
    }
    __ppl_tensor_info pe_rope_tile_2 = {.shape = {1, 32, 1, 64} , .stride = pe_rope_tile.stride, .addr = pe_rope_tile.addr + ((0) * pe_rope_tile.stride.c+(0) * pe_rope_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_rope_tile.stride.c+(0) * pe_rope_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_rope_tile.default_stride};
    __ppl_tensor_info pe_score_tile_1 = {.shape = {1, 32, 1, 64} , .stride = pe_score_tile.stride, .addr = pe_score_tile.addr + ((0) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * pe_score_tile.stride.c+(0) * pe_score_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = pe_score_tile.default_stride};
    tpu_bdc_cpy(pe_score_tile_1.addr, pe_rope_tile_2.addr, &pe_score_tile_1.shape, (pe_score_tile_1.default_stride ? NULL : &pe_score_tile_1.stride), (pe_rope_tile_2.default_stride ? NULL : &pe_rope_tile_2.stride), DT_BFP16);
    {
    scalar_t score_rope_tile_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(score_rope_tile.addr, score_rope_tile_scalar_f32, &score_rope_tile.shape, (score_rope_tile.default_stride ? NULL : &score_rope_tile.stride), DT_FP32);
    }
    tpu_bdc_fp_mm_R_trans(score_rope_tile.addr, q_rope.addr, pe_score_tile.addr, 8, 64, 32, DT_FP32, DT_BFP16);
    tpu_bdc_fp_add( score_tile.addr, score_latent_tile.addr, score_rope_tile.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_latent_tile.default_stride ? NULL : &score_latent_tile.stride), (score_rope_tile.default_stride ? NULL : &score_rope_tile.stride), DT_FP32);
    __ppl_tensor_info MASK = {.shape = {1, 8, 1, 32} , .stride = {1536, 192, 192, 1} , .addr = v31.addr + ((0) * 192+((n_1 * 32)) * 1 ) * 4, .dtype = DT_FP32, .mode = 2, .size = 1, .offset = ((0) * 192+((n_1 * 32)) * 1 ) * 4, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info mask_tile_1 = {.shape = {1, 8, 1, 32} , .stride = mask_tile.stride, .addr = mask_tile.addr + ((0) * mask_tile.stride.c+(0) * mask_tile.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * mask_tile.stride.c+(0) * mask_tile.stride.w ) * 4, .unsigned_flag = 0, .default_stride = mask_tile.default_stride};
    tpu_gdma_cpy_S2L(mask_tile_1.addr, MASK.addr, &mask_tile_1.shape, (mask_tile_1.default_stride ? NULL : &mask_tile_1.stride), (MASK.default_stride ? NULL : &MASK.stride), DT_FP32);
    tpu_bdc_fp_add( score_tile.addr, score_tile.addr, mask_tile.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), (mask_tile.default_stride ? NULL : &mask_tile.stride), DT_FP32);
    __ppl_tensor_info scores_max_1 = {.shape = {1, 8, 1, 1} , .stride = scores_max.stride, .addr = scores_max.addr + ((0) * scores_max.stride.c+(0) * scores_max.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * scores_max.stride.c+(0) * scores_max.stride.w ) * 4, .unsigned_flag = 0, .default_stride = scores_max.default_stride};
    __ppl_tensor_info scores_max_prev_1 = {.shape = {1, 8, 1, 1} , .stride = scores_max_prev.stride, .addr = scores_max_prev.addr + ((0) * scores_max_prev.stride.c+(0) * scores_max_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * scores_max_prev.stride.c+(0) * scores_max_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = scores_max_prev.default_stride};
    tpu_bdc_cpy(scores_max_prev_1.addr, scores_max_1.addr, &scores_max_prev_1.shape, (scores_max_prev_1.default_stride ? NULL : &scores_max_prev_1.stride), (scores_max_1.default_stride ? NULL : &scores_max_1.stride), DT_FP32);
    {
    scalar_t score_tile_with_prev_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(score_tile_with_prev.addr, score_tile_with_prev_scalar_f32, &score_tile_with_prev.shape, (score_tile_with_prev.default_stride ? NULL : &score_tile_with_prev.stride), DT_FP32);
    }
    __ppl_tensor_info scores_max_prev_2 = {.shape = {1, 8, 1, 1} , .stride = scores_max_prev.stride, .addr = scores_max_prev.addr + ((0) * scores_max_prev.stride.c+(0) * scores_max_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * scores_max_prev.stride.c+(0) * scores_max_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = scores_max_prev.default_stride};
    __ppl_tensor_info score_tile_with_prev_1 = {.shape = {1, 8, 1, 1} , .stride = score_tile_with_prev.stride, .addr = score_tile_with_prev.addr + ((0) * score_tile_with_prev.stride.c+(0) * score_tile_with_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_with_prev.stride.c+(0) * score_tile_with_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_with_prev.default_stride};
    tpu_bdc_cpy(score_tile_with_prev_1.addr, scores_max_prev_2.addr, &score_tile_with_prev_1.shape, (score_tile_with_prev_1.default_stride ? NULL : &score_tile_with_prev_1.stride), (scores_max_prev_2.default_stride ? NULL : &scores_max_prev_2.stride), DT_FP32);
    __ppl_tensor_info score_tile_1 = {.shape = {1, 8, 1, 32} , .stride = score_tile.stride, .addr = score_tile.addr + ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile.default_stride};
    __ppl_tensor_info score_tile_with_prev_2 = {.shape = {1, 8, 1, 32} , .stride = score_tile_with_prev.stride, .addr = score_tile_with_prev.addr + ((0) * score_tile_with_prev.stride.c+(1) * score_tile_with_prev.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile_with_prev.stride.c+(1) * score_tile_with_prev.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile_with_prev.default_stride};
    tpu_bdc_cpy(score_tile_with_prev_2.addr, score_tile_1.addr, &score_tile_with_prev_2.shape, (score_tile_with_prev_2.default_stride ? NULL : &score_tile_with_prev_2.stride), (score_tile_1.default_stride ? NULL : &score_tile_1.stride), DT_FP32);
    {
    scalar_t scores_max_scalar_f32 = {.f32 = (-1.0f / 0.0f)};
    tpu_bdc_set_C(scores_max.addr, scores_max_scalar_f32, &scores_max.shape, (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
    }
    __ppl_tensor_info tmp_buffer_max = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 32768, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_max.stride, 0, &tmp_buffer_max.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 64;
      scalar_t pad_val = {.u32 = FP_NEG_MAX(DT_FP32)};
      dim4 in_reduce_h = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      if (align_w > score_tile_with_prev.shape.w && align_w == eu_num) {
        dim4 padded_stride = {64, align_w, align_w, 1};
        dim4 padded_shape = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, score_tile_with_prev.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile_with_prev.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      } else {
        if (align_w > score_tile_with_prev.shape.w) {
          dim4 fill_shape = {score_tile_with_prev.shape.n, score_tile_with_prev.shape.c, 1, align_w - score_tile_with_prev.shape.w};
          int elem_size = 4;
          int offset = score_tile_with_prev.shape.w * elem_size;
          dim4 fill_tensor_stride = {64, align_w, score_tile_with_prev.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile_with_prev.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile_with_prev.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_max_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, pad_val);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_max.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        pad_val.u32 = FP_NEG_MAX(DT_FP32);
        tpu_bdc_fp_max_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, pad_val);
      }
    }
    tpu_bdc_fp_mul_C( prev_scaled.addr, scores_max_prev.addr, (scalar_t){.f32 = 0.0721688}, &prev_scaled.shape, (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (scores_max_prev.default_stride ? NULL : &scores_max_prev.stride), DT_FP32);
    tpu_bdc_fp_mul_C( curr_scaled.addr, scores_max.addr, (scalar_t){.f32 = 0.0721688}, &curr_scaled.shape, (curr_scaled.default_stride ? NULL : &curr_scaled.stride), (scores_max.default_stride ? NULL : &scores_max.stride), DT_FP32);
    tpu_bdc_fp_sub( scores_scale.addr, prev_scaled.addr, curr_scaled.addr, &scores_scale.shape, (scores_scale.default_stride ? NULL : &scores_scale.stride), (prev_scaled.default_stride ? NULL : &prev_scaled.stride), (curr_scaled.default_stride ? NULL : &curr_scaled.stride), DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff.addr);
    tpu_bdc_load_fp32_exp_table(table.addr);
    tpu_bdc_fp32_exp(scores_scale.addr, scores_scale.addr, work_scale0.addr, work_scale1.addr, coeff.addr, table.addr, &scores_scale.shape);
    dim4 scores_max_stride;
    __ppl_static_aligned_stride(&scores_max_stride, &scores_max.shape, DT_FP32);
    scores_max_stride.w = 0;
    tpu_bdc_fp_sub( score_tile.addr, score_tile.addr, scores_max.addr, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), &scores_max_stride, DT_FP32);
    tpu_bdc_fp_mul_C( score_tile.addr, score_tile.addr, (scalar_t){.f32 = 0.0721688}, &score_tile.shape, (score_tile.default_stride ? NULL : &score_tile.stride), (score_tile.default_stride ? NULL : &score_tile.stride), DT_FP32);
    tpu_bdc_load_fp32_exp_coeff(coeff.addr);
    tpu_bdc_load_fp32_exp_table(table.addr);
    tpu_bdc_fp32_exp(score_tile.addr, score_tile.addr, work_score0.addr, work_score1.addr, coeff.addr, table.addr, &score_tile.shape);
    __ppl_tensor_info tmp_buffer_sum = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&tmp_buffer_sum.stride, 0, &tmp_buffer_sum.shape, DT_FP32);
    {
      int eu_num = 32;
      int align_w = 32;
      scalar_t pad_val = {.f32 = 0};
      dim4 in_reduce_h = {score_tile.shape.n, score_tile.shape.c, align_w / eu_num, eu_num};
      dim4 out_reduce_h = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 in_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, eu_num};
      dim4 out_reduce_w = {score_tile.shape.n, score_tile.shape.c, 1, 1};
      dim2 kernel = {align_w / eu_num, 1};
      padding_t pad = {0, 0, 0, 0};
      dim2 stride = {1, 1};
      dim2 dilation = {1, 1};
      scalar_t scale = {.f32 = (float)1.000000000e+00};
      if (align_w > score_tile.shape.w && align_w == eu_num) {
        dim4 padded_stride = {32, align_w, align_w, 1};
        dim4 padded_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w};
        dim4 copy_shape = {score_tile.shape.n, score_tile.shape.c, 1, score_tile.shape.w};
        __ppl_tensor_info padded_input = {.shape = padded_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info input_copy = {.shape = copy_shape, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info padded_input_copy = {.shape = copy_shape, .stride = padded_stride, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = false};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_set_C(padded_input.addr, pad_val, &padded_shape, &padded_input.stride, DT_FP32);
        tpu_bdc_cpy(padded_input_copy.addr, input_copy.addr, &copy_shape, &padded_input_copy.stride, (input_copy.default_stride ? NULL : &input_copy.stride), DT_FP32);
        dim2 kernel2 = {1, eu_num};
        tpu_bdc_fp_avg_pool2d(output_view.addr, padded_input.addr, &padded_shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      } else {
        if (align_w > score_tile.shape.w) {
          dim4 fill_shape = {score_tile.shape.n, score_tile.shape.c, 1, align_w - score_tile.shape.w};
          int elem_size = 4;
          int offset = score_tile.shape.w * elem_size;
          dim4 fill_tensor_stride = {32, align_w, score_tile.shape.w, 1};
          __ppl_tensor_info fill_tensor = {.shape = fill_shape, .stride = fill_tensor_stride, .addr = score_tile.addr + offset, .dtype = DT_FP32, .mode = 0, .align_mode = 4, .size = 1, .offset = offset, .unsigned_flag = 0, .default_stride = false};
          tpu_bdc_set_C(fill_tensor.addr, pad_val, &fill_shape, (fill_tensor.default_stride ? NULL : &fill_tensor.stride), DT_FP32);
        }
        __ppl_tensor_info input_view = {.shape = in_reduce_h, .stride = {0}, .addr = score_tile.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view = {.shape = out_reduce_h, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(tmp_view.addr, input_view.addr, &input_view.shape, &kernel, &pad, &stride, &dilation, DT_FP32, scale);
        dim2 kernel2 = {1, eu_num};
        __ppl_tensor_info output_view = {.shape = out_reduce_w, .stride = {0}, .addr = scores_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        __ppl_tensor_info tmp_view2 = {.shape = in_reduce_w, .stride = {0}, .addr = tmp_buffer_sum.addr, .dtype = DT_FP32, .mode = 0, .align_mode = 1, .size = 1, .offset = 0, .unsigned_flag = 0, .default_stride = true};
        tpu_bdc_fp_avg_pool2d(output_view.addr, tmp_view2.addr, &tmp_view2.shape, &kernel2, &pad, &stride, &dilation, DT_FP32, scale);
      }
    }
    tpu_bdc_fp_mul( logsum.addr, logsum.addr, scores_scale.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_scale.default_stride ? NULL : &scores_scale.stride), DT_FP32);
    tpu_bdc_fp_add( logsum.addr, logsum.addr, scores_sum.addr, &logsum.shape, (logsum.default_stride ? NULL : &logsum.stride), (logsum.default_stride ? NULL : &logsum.stride), (scores_sum.default_stride ? NULL : &scores_sum.stride), DT_FP32);
    dim4 scores_scale_stride;
    __ppl_static_aligned_stride(&scores_scale_stride, &scores_scale.shape, DT_FP32);
    scores_scale_stride.w = 0;
    tpu_bdc_fp_mul( ctx_acc.addr, ctx_acc.addr, scores_scale.addr, &ctx_acc.shape, (ctx_acc.default_stride ? NULL : &ctx_acc.stride), (ctx_acc.default_stride ? NULL : &ctx_acc.stride), &scores_scale_stride, DT_FP32);
    __ppl_tensor_info score_tile_2 = {.shape = {1, 8, 1, 32} , .stride = score_tile.stride, .addr = score_tile.addr + ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * score_tile.stride.c+(0) * score_tile.stride.w ) * 4, .unsigned_flag = 0, .default_stride = score_tile.default_stride};
    __ppl_tensor_info prob_tile_1 = {.shape = {1, 8, 1, 32} , .stride = prob_tile.stride, .addr = prob_tile.addr + ((0) * prob_tile.stride.c+(0) * prob_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * prob_tile.stride.c+(0) * prob_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = prob_tile.default_stride};
    tpu_bdc_cast(prob_tile_1.addr, score_tile_2.addr, &prob_tile_1.shape, (prob_tile_1.default_stride ? NULL : &prob_tile_1.stride), (score_tile_2.default_stride ? NULL : &score_tile_2.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info KVcache_OUT_2 = {.shape = {1, 32, 1, 512} , .stride = {98304, 512, 512, 1} , .addr = v33.addr + (((n_1 * 32)) * 512+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((n_1 * 32)) * 512+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    __ppl_tensor_info kv_value_tile_1 = {.shape = {1, 32, 1, 512} , .stride = kv_value_tile.stride, .addr = kv_value_tile.addr + ((0) * kv_value_tile.stride.c+(0) * kv_value_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * kv_value_tile.stride.c+(0) * kv_value_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = kv_value_tile.default_stride};
    tpu_gdma_cpy_S2L(kv_value_tile_1.addr, KVcache_OUT_2.addr, &kv_value_tile_1.shape, (kv_value_tile_1.default_stride ? NULL : &kv_value_tile_1.stride), (KVcache_OUT_2.default_stride ? NULL : &KVcache_OUT_2.stride), DT_BFP16);
    tpu_bdc_fp_mm(ctx_acc.addr, prob_tile.addr, kv_value_tile.addr, 8, 32, 512, DT_FP32, DT_BFP16, true);
  }
  dim4 logsum_stride;
  __ppl_static_aligned_stride(&logsum_stride, &logsum.shape, DT_FP32);
  logsum_stride.w = 0;
  tpu_bdc_fp_div( ctx_acc.addr, ctx_acc.addr, logsum.addr, &ctx_acc.shape, (ctx_acc.default_stride ? NULL : &ctx_acc.stride), (ctx_acc.default_stride ? NULL : &ctx_acc.stride), &logsum_stride, DT_FP32);
  __ppl_tensor_info ctx_acc_1 = {.shape = {1, 8, 1, 512} , .stride = ctx_acc.stride, .addr = ctx_acc.addr + ((0) * ctx_acc.stride.c+(0) * ctx_acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * ctx_acc.stride.c+(0) * ctx_acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = ctx_acc.default_stride};
  __ppl_tensor_info ctx_cast_1 = {.shape = {1, 8, 1, 512} , .stride = ctx_cast.stride, .addr = ctx_cast.addr + ((0) * ctx_cast.stride.c+(0) * ctx_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_cast.stride.c+(0) * ctx_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_cast.default_stride};
  tpu_bdc_cast(ctx_cast_1.addr, ctx_acc_1.addr, &ctx_cast_1.shape, (ctx_cast_1.default_stride ? NULL : &ctx_cast_1.stride), (ctx_acc_1.default_stride ? NULL : &ctx_acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
  for (int n_2 = 0; n_2 < 4; ++n_2) {
    __ppl_tensor_info out_part = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 64, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_part.stride, 0, &out_part.shape, DT_FP32);
    __ppl_tensor_info out_cast = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_cast.stride, 0, &out_cast.shape, DT_BFP16);
    {
    scalar_t out_part_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(out_part.addr, out_part_scalar_f32, &out_part.shape, (out_part.default_stride ? NULL : &out_part.stride), DT_FP32);
    }
    for (int d_2 = 0; d_2 < 16; ++d_2) {
      __ppl_tensor_info ctx_chunk = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 16512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&ctx_chunk.stride, 0, &ctx_chunk.shape, DT_BFP16);
      __ppl_tensor_info wukv_v_fp8 = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_v_fp8.stride, 0, &wukv_v_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info wukv_v_scale = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_v_scale.stride, 0, &wukv_v_scale.shape, DT_BFP16);
      __ppl_tensor_info wukv_v_tile = {.shape = { 1, 32, 1, 32}, .stride = {0}, .addr = 0, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 64, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&wukv_v_tile.stride, 0, &wukv_v_tile.shape, DT_BFP16);
      __ppl_tensor_info out_sub = {.shape = { 1, 8, 1, 32}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&out_sub.stride, 0, &out_sub.shape, DT_FP32);
      __ppl_tensor_info ctx_cast_2 = {.shape = {1, 8, 1, 32} , .stride = ctx_cast.stride, .addr = ctx_cast.addr + ((0) * ctx_cast.stride.c+((d_2 * 32)) * ctx_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_cast.stride.c+((d_2 * 32)) * ctx_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_cast.default_stride};
      __ppl_tensor_info ctx_chunk_1 = {.shape = {1, 8, 1, 32} , .stride = ctx_chunk.stride, .addr = ctx_chunk.addr + ((0) * ctx_chunk.stride.c+(0) * ctx_chunk.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * ctx_chunk.stride.c+(0) * ctx_chunk.stride.w ) * 2, .unsigned_flag = 0, .default_stride = ctx_chunk.default_stride};
      tpu_bdc_cpy(ctx_chunk_1.addr, ctx_cast_2.addr, &ctx_chunk_1.shape, (ctx_chunk_1.default_stride ? NULL : &ctx_chunk_1.stride), (ctx_cast_2.default_stride ? NULL : &ctx_cast_2.stride), DT_BFP16);
      __ppl_tensor_info WUKV_1 = {.shape = {1, 32, 1, 32} , .stride = {131072, 512, 512, 1} , .addr = v22.addr + ((((n_2 * 32) + 128)) * 512+((d_2 * 32)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((((n_2 * 32) + 128)) * 512+((d_2 * 32)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_v_fp8_1 = {.shape = {1, 32, 1, 32} , .stride = wukv_v_fp8.stride, .addr = wukv_v_fp8.addr + ((0) * wukv_v_fp8.stride.c+(0) * wukv_v_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_v_fp8.stride.c+(0) * wukv_v_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_v_fp8.default_stride};
      tpu_gdma_cpy_S2L(wukv_v_fp8_1.addr, WUKV_1.addr, &wukv_v_fp8_1.shape, (wukv_v_fp8_1.default_stride ? NULL : &wukv_v_fp8_1.stride), (WUKV_1.default_stride ? NULL : &WUKV_1.stride), DT_FP8E4M3);
      __ppl_tensor_info wukv_v_fp8_2 = {.shape = {1, 32, 1, 32} , .stride = wukv_v_fp8.stride, .addr = wukv_v_fp8.addr + ((0) * wukv_v_fp8.stride.c+(0) * wukv_v_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * wukv_v_fp8.stride.c+(0) * wukv_v_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = wukv_v_fp8.default_stride};
      __ppl_tensor_info wukv_v_tile_1 = {.shape = {1, 32, 1, 32} , .stride = wukv_v_tile.stride, .addr = wukv_v_tile.addr + ((0) * wukv_v_tile.stride.c+(0) * wukv_v_tile.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_v_tile.stride.c+(0) * wukv_v_tile.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_v_tile.default_stride};
      tpu_bdc_cast(wukv_v_tile_1.addr, wukv_v_fp8_2.addr, &wukv_v_tile_1.shape, (wukv_v_tile_1.default_stride ? NULL : &wukv_v_tile_1.stride), (wukv_v_fp8_2.default_stride ? NULL : &wukv_v_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info WUKV_SCALE_1 = {.shape = {1, 32, 1, 32} , .stride = {131072, 512, 512, 1} , .addr = v24.addr + ((((n_2 * 32) + 128)) * 512+((d_2 * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((((n_2 * 32) + 128)) * 512+((d_2 * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info wukv_v_scale_1 = {.shape = {1, 32, 1, 32} , .stride = wukv_v_scale.stride, .addr = wukv_v_scale.addr + ((0) * wukv_v_scale.stride.c+(0) * wukv_v_scale.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * wukv_v_scale.stride.c+(0) * wukv_v_scale.stride.w ) * 2, .unsigned_flag = 0, .default_stride = wukv_v_scale.default_stride};
      tpu_gdma_cpy_S2L(wukv_v_scale_1.addr, WUKV_SCALE_1.addr, &wukv_v_scale_1.shape, (wukv_v_scale_1.default_stride ? NULL : &wukv_v_scale_1.stride), (WUKV_SCALE_1.default_stride ? NULL : &WUKV_SCALE_1.stride), DT_BFP16);
      tpu_bdc_fp_mul( wukv_v_tile.addr, wukv_v_tile.addr, wukv_v_scale.addr, &wukv_v_tile.shape, (wukv_v_tile.default_stride ? NULL : &wukv_v_tile.stride), (wukv_v_tile.default_stride ? NULL : &wukv_v_tile.stride), (wukv_v_scale.default_stride ? NULL : &wukv_v_scale.stride), DT_BFP16);
      {
      scalar_t out_sub_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(out_sub.addr, out_sub_scalar_f32, &out_sub.shape, (out_sub.default_stride ? NULL : &out_sub.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(out_sub.addr, ctx_chunk.addr, wukv_v_tile.addr, 8, 32, 32, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( out_part.addr, out_part.addr, out_sub.addr, &out_part.shape, (out_part.default_stride ? NULL : &out_part.stride), (out_part.default_stride ? NULL : &out_part.stride), (out_sub.default_stride ? NULL : &out_sub.stride), DT_FP32);
    }
    __ppl_tensor_info out_part_1 = {.shape = {1, 8, 1, 32} , .stride = out_part.stride, .addr = out_part.addr + ((0) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * out_part.stride.c+(0) * out_part.stride.w ) * 4, .unsigned_flag = 0, .default_stride = out_part.default_stride};
    __ppl_tensor_info out_cast_1 = {.shape = {1, 8, 1, 32} , .stride = out_cast.stride, .addr = out_cast.addr + ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast.default_stride};
    tpu_bdc_cast(out_cast_1.addr, out_part_1.addr, &out_cast_1.shape, (out_cast_1.default_stride ? NULL : &out_cast_1.stride), (out_part_1.default_stride ? NULL : &out_part_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_cast_2 = {.shape = {1, 8, 1, 32} , .stride = out_cast.stride, .addr = out_cast.addr + ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_cast.stride.c+(0) * out_cast.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_cast.default_stride};
    __ppl_tensor_info OUT = {.shape = {1, 8, 1, 32} , .stride = {1024, 128, 128, 1} , .addr = v32.addr + ((0) * 128+((n_2 * 32)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 128+((n_2 * 32)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(OUT.addr, out_cast_2.addr, &OUT.shape, (OUT.default_stride ? NULL : &OUT.stride), (out_cast_2.default_stride ? NULL : &out_cast_2.stride), DT_BFP16);
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
  global_addr_t v12;
  global_addr_t v13;
  global_addr_t v14;
  global_addr_t v15;
  global_addr_t v16;
  global_addr_t v17;
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
    api->v13,
    api->v14,
    api->v15,
    api->v16,
    api->v17);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
