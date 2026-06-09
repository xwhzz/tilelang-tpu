#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void main_kernel(fp16 *ptr_input, fp16 *ptr_output) {
  int M = 64;
  int N = 64;
  int block_m = 64;
  float eps = 1e-12f;

  dim4 global_shape = {1, M, 1, N};
  auto g_in = gtensor<fp16>(global_shape, GLOBAL, ptr_input);
  auto g_out = gtensor<fp16>(global_shape, GLOBAL, ptr_output);

  dim4 block_shape = {1, block_m, 1, N};
  dim4 avg_block_shape = {1, block_m, 1, 1};

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {
    int m = min(block_m, M - idx_m);
    dim4 in_shape = {1, m, 1, N};
    dim4 in_offset = {0, idx_m, 0, 0};
    dim4 avg_shape = {1, m, 1, 1};

    auto in_local = make_tensor<fp16>(block_shape, in_shape);
    dma::load(in_local, g_in.sub_view(in_shape, in_offset));
    auto in_fp32 = make_tensor<fp32>(block_shape, in_shape);
    tiu::cast(in_fp32, in_local);

    auto in_sq = make_tensor<fp32>(block_shape, in_shape);
    tiu::fmul(in_sq, in_fp32, in_fp32);
    auto avg_buf = make_tensor<fp32>(avg_block_shape, avg_shape);
    quick_pooling(avg_buf, in_sq, &block_shape, &in_shape, 0, 1, 1.0f / N);
    tiu::fadd(avg_buf, avg_buf, eps);

    auto mu = make_tensor<fp32>(avg_block_shape, avg_shape);
    tiu::frsqrt(mu, avg_buf, 4);
    tiu::fmul(in_fp32, in_fp32, mu);

    auto out_cast = make_tensor<fp16>(block_shape, in_shape);
    tiu::cast(out_cast, in_fp32);
    dma::store(g_out.sub_view(in_shape, in_offset), out_cast);
  }
}
