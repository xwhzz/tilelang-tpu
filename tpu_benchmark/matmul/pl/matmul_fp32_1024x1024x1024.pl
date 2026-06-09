#include "ppl.h"

using namespace ppl;

__KERNEL__ void main_kernel(fp32 *ptr_a, fp32 *ptr_b, fp32 *ptr_c) {
  int M = 1024;
  int K = 1024;
  int N = 1024;
  int block_m = 32;
  int block_k = 32;
  int block_n = 32;

  dim4 a_global_shape = {1, M, 1, K};
  dim4 b_global_shape = {1, K, 1, N};
  dim4 c_global_shape = {1, M, 1, N};
  auto g_a = gtensor<fp32>(a_global_shape, GLOBAL, ptr_a);
  auto g_b = gtensor<fp32>(b_global_shape, GLOBAL, ptr_b);
  auto g_c = gtensor<fp32>(c_global_shape, GLOBAL, ptr_c);

  dim4 a_block_shape = {1, block_m, 1, block_k};
  dim4 b_block_shape = {1, block_k, 1, block_n};
  dim4 c_block_shape = {1, block_m, 1, block_n};

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {
    int m = min(block_m, M - idx_m);
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {
      int n = min(block_n, N - idx_n);
      dim4 c_shape = {1, m, 1, n};
      dim4 c_offset = {0, idx_m, 0, idx_n};
      auto c_local = make_tensor<fp32>(c_block_shape, c_shape);
      tiu::fill(c_local, 0.0f);

      for (int idx_k = 0; idx_k < K; idx_k += block_k) {
        ppl::enable_pipeline();
        int k = min(block_k, K - idx_k);
        dim4 a_shape = {1, m, 1, k};
        dim4 a_offset = {0, idx_m, 0, idx_k};
        dim4 b_shape = {1, k, 1, n};
        dim4 b_offset = {0, idx_k, 0, idx_n};
        auto a_local = make_tensor<fp32>(a_block_shape, a_shape);
        dma::load(a_local, g_a.sub_view(a_shape, a_offset));
        auto a_fp16 = make_tensor<fp16>(a_block_shape, a_shape);
        tiu::cast(a_fp16, a_local);
        auto b_local = make_tensor<fp32>(b_block_shape, b_shape);
        dma::load(b_local, g_b.sub_view(b_shape, b_offset));
        auto b_fp16 = make_tensor<fp16>(b_block_shape, b_shape);
        tiu::cast(b_fp16, b_local);
        tiu::fmm2(c_local, a_fp16, b_fp16, true, DT_FP32);
      }
      auto c_out = make_tensor<fp32>(c_block_shape, c_shape);
      tiu::move(c_out, c_local);
      dma::store(g_c.sub_view(c_shape, c_offset), c_out);
    }
  }
}
