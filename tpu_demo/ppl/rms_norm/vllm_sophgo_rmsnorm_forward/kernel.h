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
  unsigned long long v1;
  unsigned long long v2;
  unsigned long long v3;
  unsigned long long v4;
  int32_t v5;
  float v6;
} tpu_kernel_api_main_kernel_t;

tpu_kernel_api_main_kernel_t fill_main_kernel_struct(unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, int32_t v5, float v6);

int main_kernel_check_mem(unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, int32_t v5, float v6);

int main_kernel_check_mem_s(tpu_kernel_api_main_kernel_t *api);

int main_kernel(unsigned long long v1, unsigned long long v2, unsigned long long v3, unsigned long long v4, int32_t v5, float v6);

#ifdef __cplusplus
}
#endif