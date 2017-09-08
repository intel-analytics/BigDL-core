#ifndef OPS_MIXPRECISION_GEMM_H
#define OPS_MIXPRECISION_GEMM_H

#include "./ops.h"
#include "./shuffle/shuffle_igemm.h"

void MixPrecisionGemm(ORDER order,
                  enum TRANSPOSE transA, enum TRANSPOSE transB,
                  int m, int n, int k,
                  int8_t *a, int lda, uint8_t *b, int ldb, int *c, int ldc, float fault_tolerance) {
#if defined(AVX512)
  shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance, kernel::avx512_igemm4x4x64::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>);
#elif defined(__AVX2__)
  shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance, kernel::igemm4xn::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>);
#else
  #ifdef INTEL_BIG_CORES // INTEL_BIG_CORES is the hint for Intel big cores but not supported with AVX2 and FMA
    //shuffle::InternalMixPrecisionGemm<4, 4, 8>(order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance);
  #else
    shuffle::InternalMixPrecisionGemm<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>(order, transA, transB, m, n, k, a, lda, b, ldb, c, ldc, fault_tolerance, kernel::sse42_igemm2x2x16::ApplyKernelWrapper<GEMM_SHUFFLE_KERNEL_M, GEMM_SHUFFLE_KERNEL_N, GEMM_SHUFFLE_KERNEL_K>);
  #endif

#endif
}

#endif
