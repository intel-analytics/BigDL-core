#include "../../base.h"
#include "../../common.h"
#if !defined(AVX512) && defined(__AVX2__)
namespace kernel {
namespace multidot4x2 {

static INLINE_SPECIFIER void INLINE_ATTRIBUTE Dot4x2Kernel(int8_t *&p_k1, int8_t *&p_k2, int8_t *&p_k3, int8_t *&p_k4,
                                                           uint8_t *&p_d1, uint8_t *&p_d2, SIMDSITYPE &c11,
                                                           SIMDSITYPE &c21, SIMDSITYPE &c31, SIMDSITYPE &c41,
                                                           SIMDSITYPE &c12, SIMDSITYPE &c22, SIMDSITYPE &c32,
                                                           SIMDSITYPE &c42) {
  SIMDSITYPE k1 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_k1));
  SIMDSITYPE k2 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_k2));
  SIMDSITYPE k3 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_k3));
  SIMDSITYPE k4 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_k4));
  SIMDSITYPE d1 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_d1));
  SIMDSITYPE d2 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(p_d2));
  c11 = ADDS_EPI16(MADD_EPI8(k1, d1), c11);
  c21 = ADDS_EPI16(MADD_EPI8(k2, d1), c21);
  c31 = ADDS_EPI16(MADD_EPI8(k3, d1), c31);
  c41 = ADDS_EPI16(MADD_EPI8(k4, d1), c41);
  c12 = ADDS_EPI16(MADD_EPI8(k1, d2), c12);
  c22 = ADDS_EPI16(MADD_EPI8(k2, d2), c22);
  c32 = ADDS_EPI16(MADD_EPI8(k3, d2), c32);
  c42 = ADDS_EPI16(MADD_EPI8(k4, d2), c42);
  p_k1 += OPERAND_WIDTH;
  p_k2 += OPERAND_WIDTH;
  p_k3 += OPERAND_WIDTH;
  p_k4 += OPERAND_WIDTH;
  p_d1 += OPERAND_WIDTH;
  p_d2 += OPERAND_WIDTH;
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE MultiReduceProcess(SIMDSITYPE &r, SIMDSITYPE &c1, SIMDSITYPE &c2) {
  const static SIMDSITYPE ones = SET1_EPI16(1);
  r = ADD_EPI16(HADD_EPI32(MADD_EPI16(c1, ones), MADD_EPI16(c2, ones)), r);
  c1 = ZEROS();
  c2 = ZEROS();
}

template <typename Reduce>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t *&p_k1, int8_t *&p_k2, int8_t *&p_k3, int8_t *&p_k4,
                                                           uint8_t *&p_d1, uint8_t *&p_d2, SIMDSITYPE &c11,
                                                           SIMDSITYPE &c21, SIMDSITYPE &c31, SIMDSITYPE &c41,
                                                           SIMDSITYPE &c12, SIMDSITYPE &c22, SIMDSITYPE &c32,
                                                           SIMDSITYPE &c42, SIMDSITYPE &r1, SIMDSITYPE &r2,
                                                           SIMDSITYPE &r3, SIMDSITYPE &r4, Reduce reduce) {
#if UNROLL_NUM == 4
  Dot4x2Kernel(p_k1, p_k2, p_k3, p_k4, p_d1, p_d2, c11, c21, c31, c41, c12, c22, c32, c42);
  Dot4x2Kernel(p_k1, p_k2, p_k3, p_k4, p_d1, p_d2, c11, c21, c31, c41, c12, c22, c32, c42);
  Dot4x2Kernel(p_k1, p_k2, p_k3, p_k4, p_d1, p_d2, c11, c21, c31, c41, c12, c22, c32, c42);
  Dot4x2Kernel(p_k1, p_k2, p_k3, p_k4, p_d1, p_d2, c11, c21, c31, c41, c12, c22, c32, c42);
#else
  for (size_t i = 0; i < UNROLL_NUM; ++i) {
    Dot4x2Kernel(p_k1, p_k2, p_k3, p_k4, p_d1, p_d2, c11, c21, c31, c41, c12, c22, c32, c42);
  }
#endif
  reduce(r1, c11, c21);
  reduce(r2, c31, c41);
  reduce(r3, c12, c22);
  reduce(r4, c32, c42);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostProcess(int *result[], SIMDSITYPE &r1, SIMDSITYPE &r2, SIMDSITYPE &r3,
                                                          SIMDSITYPE &r4) {
  /*
  step 0:
  r1: a1, a2, b1, b2, a3, a4, b3, b4;
  r2: c1, c2, d1, d2, c3, c4, d3, d4;
  r3: e1, e2, f1, f2, e3, e4, f3, f4;
  r4: g1, g2, h1, h2, g3, g4, h3, h4;
  step 1:
  r1: a1 + a2, b1 + b2, c1 + c2, d1 + d2, a3 + a4, b3 + b4, c3 + c4, d3 + d4
  r2: e1 + e2, f1 + f2, g1 + g2, h1 + h2, e3 + e4, f3 + f4, g3 + g4, h3 + h4
  step 2:
  r1: a1 + ... + a4, b1 + ... + b4, c1 + ... + c4, d1 + ... + d4
  r2: e1 + ... + e4, f1 + ... + f4, g1 + ... + g4, h1 + ... + h4
  */
  SIMDSITYPE result1 = HADD_EPI32(r1, r2);
  SIMDSITYPE result2 = HADD_EPI32(r3, r4);
  result1 = ADD_EPI32(PERMUTE_SI128(result1, result1, 3), result1);
  result2 = ADD_EPI32(PERMUTE_SI128(result2, result2, 3), result2);
  SIMDSITYPEHALF result1_first_half = EXTRACT_SI128(result1, 0);
  SIMDSITYPEHALF result2_first_half = EXTRACT_SI128(result2, 0);

  for (size_t i = 0; i < 4; ++i) {
    if (result[i] != NULL) {
      *(result[i]) = EXTRACT_EPI32_HALF(result1_first_half, i);
    }
    if (result[4 + i] != NULL) {
      *(result[4 + i]) = EXTRACT_EPI32_HALF(result2_first_half, i);
    }
  }
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostProcess(float *result[], SIMDSITYPE &r1, SIMDSITYPE &r2,
                                                          SIMDSITYPE &r3, SIMDSITYPE &r4, float ratio_a[],
                                                          float sum_a[], float ratio_b[], float min_b[],
                                                          bool is_block) {
  /*
  step 0:
  r1: a1, a2, b1, b2, a3, a4, b3, b4;
  r2: c1, c2, d1, d2, c3, c4, d3, d4;
  r3: e1, e2, f1, f2, e3, e4, f3, f4;
  r4: g1, g2, h1, h2, g3, g4, h3, h4;
  step 1:
  r1: a1 + a2, b1 + b2, c1 + c2, d1 + d2, a3 + a4, b3 + b4, c3 + c4, d3 + d4
  r2: e1 + e2, f1 + f2, g1 + g2, h1 + h2, e3 + e4, f3 + f4, g3 + g4, h3 + h4
  step 2:
  r1: a1 + ... + a4, b1 + ... + b4, c1 + ... + c4, d1 + ... + d4
  r2: e1 + ... + e4, f1 + ... + f4, g1 + ... + g4, h1 + ... + h4
  */
  SIMDSITYPE result1 = HADD_EPI32(r1, r2);
  SIMDSITYPE result2 = HADD_EPI32(r3, r4);
  result1 = ADD_EPI32(PERMUTE_SI128(result1, result1, 3), result1);
  result2 = ADD_EPI32(PERMUTE_SI128(result2, result2, 3), result2);
  SIMDSITYPEHALF result1_first_half = EXTRACT_SI128(result1, 0);
  SIMDSITYPEHALF result2_first_half = EXTRACT_SI128(result2, 0);
  if (!is_block) {
    for (size_t i = 0; i < 4; ++i) {
      if (result[i] != NULL) {
        *(result[i]) = ratio_a[i] * ratio_b[0] * EXTRACT_EPI32_HALF(result1_first_half, i) + sum_a[i] * min_b[0];
      }
      if (result[4 + i] != NULL) {
        *(result[4 + i]) = ratio_a[i] * ratio_b[1] * EXTRACT_EPI32_HALF(result2_first_half, i) + sum_a[i] * min_b[1];
      }
    }
  } else {
    SIMDPSTYPEHALF simd_ratio_a = LOADU_PS_HALF(ratio_a);
    SIMDPSTYPEHALF simd_sum_a = LOADU_PS_HALF(sum_a);
    SIMDPSTYPEHALF simd_ratio_b0 = SET1_PS_HALF(ratio_b[0]);
    SIMDPSTYPEHALF simd_min_b0 = SET1_PS_HALF(min_b[0]);
    SIMDPSTYPEHALF simd_ratio_b1 = SET1_PS_HALF(ratio_b[1]);
    SIMDPSTYPEHALF simd_min_b1 = SET1_PS_HALF(min_b[1]);
    SIMDPSTYPEHALF final_result1 =
        FMA_PS_HALF(MUL_PS_HALF(simd_ratio_a, simd_ratio_b0), EPI32TOPS_HALF(result1_first_half),
                    MUL_PS_HALF(simd_sum_a, simd_min_b0));
    SIMDPSTYPEHALF final_result2 =
        FMA_PS_HALF(MUL_PS_HALF(simd_ratio_a, simd_ratio_b1), EPI32TOPS_HALF(result2_first_half),
                    MUL_PS_HALF(simd_sum_a, simd_min_b1));
    *(result[0]) = CVTSS_PS_HALF(final_result1);
    *(result[1]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result1, 1));
    *(result[2]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result1, 2));
    *(result[3]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result1, 3));
    *(result[4]) = CVTSS_PS_HALF(final_result2);
    *(result[5]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result2, 1));
    *(result[6]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result2, 2));
    *(result[7]) = CVTSS_PS_HALF(PERMUTE_PS_HALF(final_result2, 3));
  }
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelDot4x2(int8_t *a[], uint8_t *b[], int *result[],
                                                                size_t length) {
  // 4 INT32 Accumulator
  SIMDSITYPE r1 = ZEROS();
  SIMDSITYPE r2 = ZEROS();
  SIMDSITYPE r3 = ZEROS();
  SIMDSITYPE r4 = ZEROS();
  // 8 INT32 Accunulator
  SIMDSITYPE c11 = ZEROS();
  SIMDSITYPE c21 = ZEROS();
  SIMDSITYPE c31 = ZEROS();
  SIMDSITYPE c41 = ZEROS();
  SIMDSITYPE c12 = ZEROS();
  SIMDSITYPE c22 = ZEROS();
  SIMDSITYPE c32 = ZEROS();
  SIMDSITYPE c42 = ZEROS();
  int8_t *a1 = a[0];
  int8_t *a2 = a[1];
  int8_t *a3 = a[2];
  int8_t *a4 = a[3];
  uint8_t *b1 = b[0];
  uint8_t *b2 = b[1];
  size_t k = length;
  while (k >= UNROLL_NUM * OPERAND_WIDTH) {
    KernelReduce(a1, a2, a3, a4, b1, b2, c11, c21, c31, c41, c12, c22, c32, c42, r1, r2, r3, r4, MultiReduceProcess);
    k -= UNROLL_NUM * OPERAND_WIDTH;
  }
  if (k >= OPERAND_WIDTH) {
    while (k >= OPERAND_WIDTH) {
      Dot4x2Kernel(a1, a2, a3, a4, b1, b2, c11, c21, c31, c41, c12, c22, c32, c42);
      k -= OPERAND_WIDTH;
    }
    MultiReduceProcess(r1, c11, c21);
    MultiReduceProcess(r2, c31, c41);
    MultiReduceProcess(r3, c12, c22);
    MultiReduceProcess(r4, c32, c42);
  }
  PostProcess(result, r1, r2, r3, r4);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelDot4x2(int8_t *a[], uint8_t *b[], float *result[],
                                                                size_t length, float ratio_a[], float sum_a[],
                                                                float ratio_b[], float min_b[], bool is_block) {
  // 4 INT32 Accumulator
  SIMDSITYPE r1 = ZEROS();
  SIMDSITYPE r2 = ZEROS();
  SIMDSITYPE r3 = ZEROS();
  SIMDSITYPE r4 = ZEROS();
  // 8 INT32 Accunulator
  SIMDSITYPE c11 = ZEROS();
  SIMDSITYPE c21 = ZEROS();
  SIMDSITYPE c31 = ZEROS();
  SIMDSITYPE c41 = ZEROS();
  SIMDSITYPE c12 = ZEROS();
  SIMDSITYPE c22 = ZEROS();
  SIMDSITYPE c32 = ZEROS();
  SIMDSITYPE c42 = ZEROS();
  int8_t *a1 = a[0];
  int8_t *a2 = a[1];
  int8_t *a3 = a[2];
  int8_t *a4 = a[3];
  uint8_t *b1 = b[0];
  uint8_t *b2 = b[1];
  size_t k = length;
  while (k >= UNROLL_NUM * OPERAND_WIDTH) {
    KernelReduce(a1, a2, a3, a4, b1, b2, c11, c21, c31, c41, c12, c22, c32, c42, r1, r2, r3, r4, MultiReduceProcess);
    k -= UNROLL_NUM * OPERAND_WIDTH;
  }
  if (k >= OPERAND_WIDTH) {
    while (k >= OPERAND_WIDTH) {
      Dot4x2Kernel(a1, a2, a3, a4, b1, b2, c11, c21, c31, c41, c12, c22, c32, c42);
      k -= OPERAND_WIDTH;
    }
    MultiReduceProcess(r1, c11, c21);
    MultiReduceProcess(r2, c31, c41);
    MultiReduceProcess(r3, c12, c22);
    MultiReduceProcess(r4, c32, c42);
  }
  PostProcess(result, r1, r2, r3, r4, ratio_a, sum_a, ratio_b, min_b, is_block);
}
}
}
#endif
