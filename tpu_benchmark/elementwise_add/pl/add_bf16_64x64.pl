#include "ppl.h"

using namespace ppl;

// Parameter order matches tilelang: (A, B, C)
__KERNEL__ void main_kernel(bf16 *ptr_a, bf16 *ptr_b, bf16 *ptr_c) {
  int C = 64;
  int W = 64;
  int block_c = 64;
  int block_w = 16;

  dim4 global_shape = {1, C, 1, W};
  auto g_a = gtensor<bf16>(global_shape, GLOBAL, ptr_a);
  auto g_b = gtensor<bf16>(global_shape, GLOBAL, ptr_b);
  auto g_c = gtensor<bf16>(global_shape, GLOBAL, ptr_c);

  dim4 block_shape = {1, block_c, 1, block_w};

  for (int idx_c = 0; idx_c < C; idx_c += block_c) {
    int c = min(block_c, C - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w) {
      ppl::enable_pipeline();
      int w = min(block_w, W - idx_w);

      dim4 in_shape = {1, c, 1, w};
      auto a = make_tensor<bf16>(block_shape, in_shape);
      auto b = make_tensor<bf16>(block_shape, in_shape);
      auto out = make_tensor<bf16>(block_shape, in_shape);

      dim4 in_offset = {0, idx_c, 0, idx_w};
      dma::load(a, g_a.sub_view(in_shape, in_offset));
      dma::load(b, g_b.sub_view(in_shape, in_offset));

      tiu::fadd(out, a, b);

      dma::store(g_c.sub_view(in_shape, in_offset), out);
    }
  }
}
