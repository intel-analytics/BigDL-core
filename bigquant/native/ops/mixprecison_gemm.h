/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPS_MIXPRECISION_GEMM_H
#define OPS_MIXPRECISION_GEMM_H

#include "./ops.h"
#include "./shuffle/shuffle_igemm.h"

void MixPrecisionGemm(ORDER order, enum TRANSPOSE transA, enum TRANSPOSE transB, int m, int n, int k, int8_t *a,
                      int lda, uint8_t *b, int ldb, int *c, int ldc, float fault_tolerance) {
#if defined(AVX512)
  // shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(order,
  // transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance,
  // kernel::avx512_igemm4x4x64::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N,
  // GEMM_SHUFFLE_KERNEL_K>);
  shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(
      order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance,
      kernel::avx512_igemm8x8x8::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N,
                                                    GEMM_SHUFFLE_KERNEL_K>);
#elif defined(__AVX2__)
  shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(
      order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance,
      kernel::igemm4xn::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>);
#else
#ifdef INTEL_BIG_CORES  // INTEL_BIG_CORES is the hint for Intel big cores but not supported with AVX2 and FMA
// shuffle::InternalMixPrecisionGemm<4, 4, 8>(order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance);
#else
  shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(
      order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance,
      kernel::sse42_igemm2x2x16::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N,
                                                    GEMM_SHUFFLE_KERNEL_K>);
#endif

#endif
}

#endif
