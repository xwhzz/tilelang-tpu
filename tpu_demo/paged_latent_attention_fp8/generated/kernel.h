#pragma once
#ifndef __bm1690__
#define __bm1690__
#endif

#include <stdint.h>
#include <assert.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned long long ptr_v1;
  unsigned long long ptr_v2;
  unsigned long long ptr_v3;
  unsigned long long ptr_v4;
  unsigned long long ptr_v5;
  unsigned long long ptr_v6;
  unsigned long long ptr_v7;
  unsigned long long ptr_v8;
  unsigned long long ptr_v9;
  unsigned long long ptr_v10;
  unsigned long long ptr_v11;
  unsigned long long ptr_v12;
  unsigned long long ptr_v13;
  unsigned long long ptr_v14;
  unsigned long long ptr_v15;
  unsigned long long ptr_v16;
  unsigned long long ptr_v17;
} tpu_kernel_api_main_kernel_t;

tpu_kernel_api_main_kernel_t fill_main_kernel_struct(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17);

int main_kernel_check_mem(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17);

int main_kernel_check_mem_s(tpu_kernel_api_main_kernel_t *api);

int main_kernel(unsigned long long ptr_v1, unsigned long long ptr_v2, unsigned long long ptr_v3, unsigned long long ptr_v4, unsigned long long ptr_v5, unsigned long long ptr_v6, unsigned long long ptr_v7, unsigned long long ptr_v8, unsigned long long ptr_v9, unsigned long long ptr_v10, unsigned long long ptr_v11, unsigned long long ptr_v12, unsigned long long ptr_v13, unsigned long long ptr_v14, unsigned long long ptr_v15, unsigned long long ptr_v16, unsigned long long ptr_v17);

#ifdef __cplusplus
}
#endif