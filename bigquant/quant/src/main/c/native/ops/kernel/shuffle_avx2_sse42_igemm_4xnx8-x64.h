#ifndef IGEMM4XN_X64
#define IGEMM4XN_X64
#include "../../base.h"
#include "../kernel-common.h"

namespace kernel {
namespace igemm4xn {
#ifdef __AVX2__
static INLINE_SPECIFIER SIMDSITYPE INLINE_ATTRIBUTE ReOrderResult(SIMDSITYPE &result, SIMDSITYPE &index) {
  return PERMUTE_EPI32(result, index);
}
#endif


template <typename kernel_function, typename reduce_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22,
  SIMDSITYPE &c31, SIMDSITYPE &c32,
  SIMDSITYPE &c41, SIMDSITYPE &c42,
  SIMDSITYPE &sum1, SIMDSITYPE &sum2,
  SIMDSITYPE &sum3, SIMDSITYPE &sum4,
  SIMDSITYPE &threshold, SIMDSITYPE &ones,
  kernel_function kernel, reduce_function reduce) {
  kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
  kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
  kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
  kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
  reduce(c11, c12, sum1, threshold, ones);
  reduce(c21, c22, sum2, threshold, ones);
  reduce(c31, c32, sum3, threshold, ones);
  reduce(c41, c42, sum4, threshold, ones);
}

#ifdef  __AVX2__
static INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX2Kernel4x8x8(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22,
  SIMDSITYPE &c31, SIMDSITYPE &c32,
  SIMDSITYPE &c41, SIMDSITYPE &c42) {
  SIMDSITYPE a = LOAD_SI256(reinterpret_cast<SIMDSITYPE*>(pa));
  SIMDSITYPE a1 = PERMUTE_EPI64(a, 0);
  SIMDSITYPE b1 = LOAD_SI256(reinterpret_cast<SIMDSITYPE*>(pb));
  SIMDSITYPE b2 = LOAD_SI256(reinterpret_cast<SIMDSITYPE*>(pb + 32));

  c11 = ADDS_EPI16(c11, MADD_EPI8(b1, a1));
  c12 = ADDS_EPI16(c12, MADD_EPI8(b2, a1));
  // 1 + (1 << 2) + (1 << 4) + (1 << 6) = 85
  SIMDSITYPE a2 = PERMUTE_EPI64(a, 85);
  c21 = ADDS_EPI16(c21, MADD_EPI8(b1, a2));
  c22 = ADDS_EPI16(c22, MADD_EPI8(b2, a2));
  // 2 + (2 << 2) + (2 << 4) + (2 << 6) = 170
  SIMDSITYPE a3 = PERMUTE_EPI64(a, 170);
  c31 = ADDS_EPI16(c31, MADD_EPI8(b1, a3));
  c32 = ADDS_EPI16(c32, MADD_EPI8(b2, a3));
  // 3 + (3 << 2) + (3 << 4) + (3 << 6) = 255
  SIMDSITYPE a4 = PERMUTE_EPI64(a, 255);
  c41 = ADDS_EPI16(c41, MADD_EPI8(b1, a4));
  c42 = ADDS_EPI16(c42, MADD_EPI8(b2, a4));
  pa += 32;
  pb += 64;
}
#else
static INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel4x4x8(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22,
  SIMDSITYPE &c31, SIMDSITYPE &c32,
  SIMDSITYPE &c41, SIMDSITYPE &c42) {
  SIMDSITYPE tmp1 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pa));
  SIMDSITYPE tmp2 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pa + 16));
  // (0 << 0) + (1 << 2) + (0 << 4) + (1 << 6) = 68
  SIMDSITYPE a1 = SHUFFLE_EPI32(tmp1, 68);
  // (2 << 0) + (3 << 2) + (2 << 4) + (3 << 6) = 238
  SIMDSITYPE a2 = SHUFFLE_EPI32(tmp1, 238);
  // (0 << 0) + (1 << 2) + (0 << 4) + (1 << 6) = 68
  SIMDSITYPE a3 = SHUFFLE_EPI32(tmp2, 68);
  // (2 << 0) + (3 << 2) + (2 << 4) + (3 << 6) = 238
  SIMDSITYPE a4 = SHUFFLE_EPI32(tmp2, 238);
  SIMDSITYPE b1 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pb));
  SIMDSITYPE b2 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pb + 16));
  c11 = ADDS_EPI16(c11, MADD_EPI8(b1, a1));
  c12 = ADDS_EPI16(c12, MADD_EPI8(b2, a1));
  c21 = ADDS_EPI16(c21, MADD_EPI8(b1, a2));
  c22 = ADDS_EPI16(c22, MADD_EPI8(b2, a2));
  c31 = ADDS_EPI16(c31, MADD_EPI8(b1, a3));
  c32 = ADDS_EPI16(c32, MADD_EPI8(b2, a3));
  c41 = ADDS_EPI16(c41, MADD_EPI8(b1, a4));
  c42 = ADDS_EPI16(c42, MADD_EPI8(b2, a4));
  pa += 32;
  pb += 32;
}
#endif

static INLINE_SPECIFIER SIMDSITYPE INLINE_ATTRIBUTE HADDSUM16TOSum32(SIMDSITYPE &localsum1, SIMDSITYPE &localsum2) {
#ifdef __AVX2__
  SIMDSITYPE lo1 = EPI16TOEPI32(EXTRACT_SI128(localsum1, 0)); // Get Lower 128bits into lo
  SIMDSITYPE hi1 = EPI16TOEPI32(EXTRACT_SI128(localsum1, 1)); // Get Higher 128bits into hi
  SIMDSITYPE lo2 = EPI16TOEPI32(EXTRACT_SI128(localsum2, 0));
  SIMDSITYPE hi2 = EPI16TOEPI32(EXTRACT_SI128(localsum2, 1));
#else
  SIMDSITYPE lo1 = EPI16TOEPI32(localsum1); // a3,a2,a1,a0
  SIMDSITYPE hi1 = EPI16TOEPI32(SRLI_SI128(localsum1, 8)); // b3,b2,b1,b0
  SIMDSITYPE lo2 = EPI16TOEPI32(localsum2); // c3,c2,c1,c0
  SIMDSITYPE hi2 = EPI16TOEPI32(SRLI_SI128(localsum2, 8)); // d3,d2,d1,d0
#endif
  localsum1 = ZEROS();
  localsum2 = ZEROS();
  SIMDSITYPE tmp1 = HADD_EPI32(lo1, hi1); // reduce
  SIMDSITYPE tmp2 = HADD_EPI32(lo2, hi2); // reduce
  return HADD_EPI32(tmp1, tmp2);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE HaddPairReduce(SIMDSITYPE &c1, SIMDSITYPE &c2, SIMDSITYPE &sum,
                          SIMDSITYPE &threshold, SIMDSITYPE &ones) {
  sum = ADD_EPI32(HADD_EPI32(MADD_EPI16(c1, ones), MADD_EPI16(c2, ones)), sum);
  c1 = ZEROS(); c2 = ZEROS();
/*
  SIMDSITYPE saturated = MAX_EPI16(ABS_EPI16(c1), ABS_EPI16(c2));
  SIMDSITYPE flag = CMP_EPI16(saturated, threshold);
#ifdef _MSC_VER
  if (TESTZ_SI(ones, flag)) {

  } else {
    sum = ADD_EPI32(sum, HADDSUM16TOSum32(c1, c2));
  }
#else
#ifdef __AVX2__
  asm goto(
      "vptest %0, %1\n"
      "jz %l2\n"
      ::"x"(ones), "x"(flag):: haddpairreduceend);
#else
  asm goto(
      "ptest %0, %1\n"
      "jz %l2\n"
      ::"x"(ones), "x"(flag):: haddpairreduceend);
#endif
  sum = ADD_EPI32(sum, HADDSUM16TOSum32(c1, c2));
  return;
haddpairreduceend:
  return;
#endif
*/
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostHaddReduce(
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22,
  SIMDSITYPE &c31, SIMDSITYPE &c32,
  SIMDSITYPE &c41, SIMDSITYPE &c42,
  SIMDSITYPE &sum1, SIMDSITYPE &sum2,
  SIMDSITYPE &sum3, SIMDSITYPE &sum4) {
  SIMDSITYPE ones = SET1_EPI16(1);
  sum1 = ADD_EPI32(HADD_EPI32(MADD_EPI16(c11, ones), MADD_EPI16(c12, ones)), sum1);
  sum2 = ADD_EPI32(HADD_EPI32(MADD_EPI16(c21, ones), MADD_EPI16(c22, ones)), sum2);
  sum3 = ADD_EPI32(HADD_EPI32(MADD_EPI16(c31, ones), MADD_EPI16(c32, ones)), sum3);
  sum4 = ADD_EPI32(HADD_EPI32(MADD_EPI16(c41, ones), MADD_EPI16(c42, ones)), sum4);

  //sum1 = ADD_EPI32(sum1, HADDSUM16TOSum32(c11, c12));
  //sum2 = ADD_EPI32(sum2, HADDSUM16TOSum32(c21, c22));
  //sum3 = ADD_EPI32(sum3, HADDSUM16TOSum32(c31, c32));
  //sum4 = ADD_EPI32(sum4, HADDSUM16TOSum32(c41, c42));
#ifdef __AVX2__
  // The following index is for hadd method
  SIMDSITYPE index = SET_EPI32(7, 6, 3, 2, 5, 4, 1, 0);
  // The following index is for pack method
  // SIMDType index = SETINT32(7, 3, 5, 1, 6, 2, 4, 0);
  sum1 = ReOrderResult(sum1, index);
  sum2 = ReOrderResult(sum2, index);
  sum3 = ReOrderResult(sum3, index);
  sum4 = ReOrderResult(sum4, index);
#endif
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitBlockResult(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4, void* result[], size_t length, size_t valid_lanes) {
  STOREU_SI(reinterpret_cast<SIMDSITYPE *>(result[0]), sum1);
  STOREU_SI(reinterpret_cast<SIMDSITYPE *>(result[1]), sum2);
  STOREU_SI(reinterpret_cast<SIMDSITYPE *>(result[2]), sum3);
  STOREU_SI(reinterpret_cast<SIMDSITYPE *>(result[3]), sum4);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitResult(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4, void* result[], size_t length, size_t valid_lanes) {
#ifdef __AVX2__
  SIMDSITYPEHALF tmp1, tmp2, tmp3, tmp4;
  tmp1 = EXTRACT_SI128(sum1, 0); tmp2 = EXTRACT_SI128(sum2, 0);
  tmp3 = EXTRACT_SI128(sum3, 0); tmp4 = EXTRACT_SI128(sum4, 0);
  for (int ky = 0; ky < valid_lanes; ++ky) {
    if (length == 4) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32_HALF(tmp1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32_HALF(tmp2, 0);
      *(reinterpret_cast<int*>(result[2]) + ky) = EXTRACT_EPI32_HALF(tmp3, 0);
      *(reinterpret_cast<int*>(result[3]) + ky) = EXTRACT_EPI32_HALF(tmp4, 0);
    } else if (length == 3) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32_HALF(tmp1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32_HALF(tmp2, 0);
      *(reinterpret_cast<int*>(result[2]) + ky) = EXTRACT_EPI32_HALF(tmp3, 0);
    } else if (length == 2) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32_HALF(tmp1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32_HALF(tmp2, 0);
    } else {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32_HALF(tmp1, 0);
    }
    if (ky == 3) {
      tmp1 = EXTRACT_SI128(sum1, 1); tmp2 = EXTRACT_SI128(sum2, 1);
      tmp3 = EXTRACT_SI128(sum3, 1); tmp4 = EXTRACT_SI128(sum4, 1);
    } else {
      tmp1 = SRLI_SI128_HALF(tmp1, 4); tmp2 = SRLI_SI128_HALF(tmp2, 4);
      tmp3 = SRLI_SI128_HALF(tmp3, 4); tmp4 = SRLI_SI128_HALF(tmp4, 4);
    }
  }
#else
  for (int ky = 0; ky < valid_lanes; ++ky) {
    if (length == 4) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32(sum2, 0);
      *(reinterpret_cast<int*>(result[2]) + ky) = EXTRACT_EPI32(sum3, 0);
      *(reinterpret_cast<int*>(result[3]) + ky) = EXTRACT_EPI32(sum4, 0);
    } else if (length == 3) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32(sum2, 0);
      *(reinterpret_cast<int*>(result[2]) + ky) = EXTRACT_EPI32(sum3, 0);
    } else if (length == 2) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum1, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32(sum2, 0);
    } else {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum1, 0);
    }
		sum1 = SRLI_SI128(sum1, 4); sum2 = SRLI_SI128(sum2, 4);
		sum3 = SRLI_SI128(sum3, 4); sum4 = SRLI_SI128(sum4, 4);
  }
#endif
}

template<size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NCHWFMABlockResult(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4,
                                                                float* result[], size_t length, size_t valid_lanes, size_t i_index, size_t j_index,
                                                                float* ratio_a, float* ratio_b, float* min_b, float* kernel_sum, float* bias,
                                                                bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                                float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  const static SIMDPSTYPE zero = ZERO_PS();
  SIMDPSTYPE result1, result2, result3, result4;
  SIMDPSTYPE bias1, bias2, bias3, bias4;
  SIMDPSTYPE simd_ratio_b = LOADU_PS(ratio_b + j_index);
  SIMDPSTYPE simd_min_b = LOADU_PS(min_b + j_index);
  SIMDPSTYPE coeffi1 = MUL_PS(SET1_PS(ratio_a[i_index]), simd_ratio_b);
  SIMDPSTYPE coeffi2 = MUL_PS(SET1_PS(ratio_a[i_index + 1]), simd_ratio_b);
  SIMDPSTYPE coeffi3 = MUL_PS(SET1_PS(ratio_a[i_index + 2]), simd_ratio_b);
  SIMDPSTYPE coeffi4 = MUL_PS(SET1_PS(ratio_a[i_index + 3]), simd_ratio_b);
  if (bias != NULL) {
    bias1 = SET1_PS(bias[i_index]);
    bias2 = SET1_PS(bias[i_index + 1]);
    bias3 = SET1_PS(bias[i_index + 2]);
    bias4 = SET1_PS(bias[i_index + 3]);
  } else {
    bias1 = ZERO_PS();
    bias2 = ZERO_PS();
    bias3 = ZERO_PS();
    bias4 = ZERO_PS();
  }
  result1 = FMA_PS(EPI32TOPS(sum1), coeffi1, FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index]), bias1));
  result2 = FMA_PS(EPI32TOPS(sum2), coeffi2, FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 1]), bias2));
  result3 = FMA_PS(EPI32TOPS(sum3), coeffi3, FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 2]), bias3));
  result4 = FMA_PS(EPI32TOPS(sum4), coeffi4, FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 3]), bias4));
  // TODO(yandai) Verify for SSE
  if (conv_relu_fusion) { // conv + relu
  } else if (conv_bn_fusion) { // conv + bn
  } else if (conv_bn_relu_fusion) { // conv + bn + relu
  } else if (conv_relu_bn_fusion) { // conv + relu + fusion
  }
  STOREU_PS(result[0 * kernel_n], result1);
  STOREU_PS(result[1 * kernel_n], result2);
  STOREU_PS(result[2 * kernel_n], result3);
  STOREU_PS(result[3 * kernel_n], result4);
}

template<size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NHWCFMABlockResult(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4,
                                                                float* result[], size_t length, size_t valid_lanes, size_t i_index, size_t j_index,
                                                                float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                                                bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                                float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  const static SIMDPSTYPE zero = ZERO_PS();
  SIMDPSTYPE bias1, bias2, bias3, bias4;
  SIMDPSTYPE simd_ratio_b = LOADU_PS(ratio_b + j_index);
  SIMDPSTYPE simd_min_b = LOADU_PS(min_b + j_index);
  SIMDPSTYPE coeffi1 = MUL_PS(SET1_PS(ratio_a[i_index]), simd_ratio_b);
  SIMDPSTYPE coeffi2 = MUL_PS(SET1_PS(ratio_a[i_index + 1]), simd_ratio_b);
  SIMDPSTYPE coeffi3 = MUL_PS(SET1_PS(ratio_a[i_index + 2]), simd_ratio_b);
  SIMDPSTYPE coeffi4 = MUL_PS(SET1_PS(ratio_a[i_index + 3]), simd_ratio_b);
  if (bias != NULL) {
    bias1 = SET1_PS(bias[i_index]);
    bias2 = SET1_PS(bias[i_index + 1]);
    bias3 = SET1_PS(bias[i_index + 2]);
    bias4 = SET1_PS(bias[i_index + 3]);
  } else {
    bias1 = ZERO_PS();
    bias2 = ZERO_PS();
    bias3 = ZERO_PS();
    bias4 = ZERO_PS();
  }
  bias1 = FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index]), bias1);
  bias2 = FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 1]), bias2);
  bias3 = FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 2]), bias3);
  bias4 = FMA_PS(simd_min_b, SET1_PS(kernel_sum[i_index + 3]), bias4);
  SIMDPSTYPE result1;
  SIMDPSTYPE result2;
  SIMDPSTYPE result3;
  SIMDPSTYPE result4;
  result1 = FMA_PS(EPI32TOPS(sum1), coeffi1, bias1); //a1,...a8
  result2 = FMA_PS(EPI32TOPS(sum2), coeffi2, bias2); //b1,...b8
  result3 = FMA_PS(EPI32TOPS(sum3), coeffi3, bias3); //c1,...c8
  result4 = FMA_PS(EPI32TOPS(sum4), coeffi4, bias4); //d1,...d8
  if (conv_relu_fusion) { // conv + relu
    PRELU(result1, zero); PRELU(result2, zero); PRELU(result3, zero); PRELU(result4, zero);
  } else if (conv_bn_fusion) { // conv + bn
    SIMDPSTYPE global_mean1 = SET1_PS(global_mean[i_index]);
    SIMDPSTYPE global_mean2 = SET1_PS(global_mean[i_index + 1]);
    SIMDPSTYPE global_mean3 = SET1_PS(global_mean[i_index + 2]);
    SIMDPSTYPE global_mean4 = SET1_PS(global_mean[i_index + 3]);
    SIMDPSTYPE mul_variance_coeff1 = SET1_PS(mul_variance_coeff[i_index]);
    SIMDPSTYPE mul_variance_coeff2 = SET1_PS(mul_variance_coeff[i_index + 1]);
    SIMDPSTYPE mul_variance_coeff3 = SET1_PS(mul_variance_coeff[i_index + 2]);
    SIMDPSTYPE mul_variance_coeff4 = SET1_PS(mul_variance_coeff[i_index + 3]);
    SIMDPSTYPE scale1 = SET1_PS((scale == NULL)? 1.0f : scale[i_index]);
    SIMDPSTYPE scale2 = SET1_PS((scale == NULL)? 1.0f : scale[i_index + 1]);
    SIMDPSTYPE scale3 = SET1_PS((scale == NULL)? 1.0f : scale[i_index + 2]);
    SIMDPSTYPE scale4 = SET1_PS((scale == NULL)? 1.0f : scale[i_index + 3]);
    SIMDPSTYPE shift1 = SET1_PS((shift == NULL)? 0.0f : shift[i_index]);
    SIMDPSTYPE shift2 = SET1_PS((shift == NULL)? 0.0f : shift[i_index + 1]);
    SIMDPSTYPE shift3 = SET1_PS((shift == NULL)? 0.0f : shift[i_index + 2]);
    SIMDPSTYPE shift4 = SET1_PS((shift == NULL)? 0.0f : shift[i_index + 3]);
    BN(result1, global_mean1, mul_variance_coeff1, scale1, scale1);
    BN(result2, global_mean2, mul_variance_coeff2, scale2, scale2);
    BN(result3, global_mean3, mul_variance_coeff3, scale3, scale3);
    BN(result4, global_mean4, mul_variance_coeff4, scale4, scale4);
  } else if (conv_bn_relu_fusion) { // conv + bn + relu

  } else if (conv_relu_bn_fusion) { // conv + relu + fusion
    PRELU(result1, zero); PRELU(result2, zero); PRELU(result3, zero); PRELU(result4, zero);
  }
	// AVX2	SSE4_2
	// a1,b1,a2,b2,a5,b5,a6,b6;	a1,b1,a2,b2
	// a3,b3,a4,b4,a7,b7,a8,b8; a3,b3,a3,b4
	// c1,d1,c2,d2,c5,d5,c6,d6; c1,d1,c2,d2
	// c3,d3,c4,d4,c7,d7,c8,d8; c3,d3,c4,d4
  SIMDPSTYPE result12lo = UNPACKLO_PS(result1, result2);
  SIMDPSTYPE result12hi = UNPACKHI_PS(result1, result2);
  SIMDPSTYPE result34lo = UNPACKLO_PS(result3, result4);
  SIMDPSTYPE result34hi = UNPACKHI_PS(result3, result4);

#ifdef _MSC_VER
  // AVX2 SSE4_2
  // a1,c1,b1,d1,a5,c5,b5,d5; a1,b1,c1,d1
  // a2,c2,b2,d2,a6,c6,b6,d8; a2,b2,c2,d2
  // a3,c3,b3,d3,a7,c7,b7,d7; a3,b3,c3,d4
  // a4,c4,b3,d4,a8,c8,b8,d8; a4,b3,c3,d4
  SIMDPSTYPE mix1 = UNPACKLO_PS(result12lo, result34lo);
  SIMDPSTYPE mix2 = UNPACKHI_PS(result12lo, result34lo);
  SIMDPSTYPE mix3 = UNPACKLO_PS(result12hi, result34hi);
  SIMDPSTYPE mix4 = UNPACKHI_PS(result12hi, result34hi);
  // 0 + (2 << 2) + (1 << 4) + (3 << 6) = 216
  mix1 = PERMUTE_PS(mix1, 216);
  mix2 = PERMUTE_PS(mix2, 216);
  mix3 = PERMUTE_PS(mix3, 216);
  mix4 = PERMUTE_PS(mix4, 216);
#else
	// AVX2 SSE4_2
	// a1,b1,c1,d1,a5,b5,c5,d5; a1,b1,c1,d1
	// a2,b2,c2,d2,a6,b6,c6,d8; a2,b2,c2,d2
	// a3,b3,c3,d3,a7,b7,c7,d7; a3,b3,c3,d4
	// a4,b4,c3,d4,a8,b8,c8,d8; a4,b3,c3,d4
  SIMDPSTYPE mix1 = reinterpret_cast<SIMDPSTYPE>(UNPACKLO_PD(reinterpret_cast<SIMDPDTYPE>(result12lo), reinterpret_cast<SIMDPDTYPE>(result34lo)));
  SIMDPSTYPE mix2 = reinterpret_cast<SIMDPSTYPE>(UNPACKHI_PD(reinterpret_cast<SIMDPDTYPE>(result12lo), reinterpret_cast<SIMDPDTYPE>(result34lo)));
  SIMDPSTYPE mix3 = reinterpret_cast<SIMDPSTYPE>(UNPACKLO_PD(reinterpret_cast<SIMDPDTYPE>(result12hi), reinterpret_cast<SIMDPDTYPE>(result34hi)));
  SIMDPSTYPE mix4 = reinterpret_cast<SIMDPSTYPE>(UNPACKHI_PD(reinterpret_cast<SIMDPDTYPE>(result12hi), reinterpret_cast<SIMDPDTYPE>(result34hi)));
#endif
#ifdef __AVX2__
  STOREU256_PS_HALF(result[0 * kernel_m], EXTRACT_PS_HALF(mix1, 0));
  STOREU256_PS_HALF(result[1 * kernel_m], EXTRACT_PS_HALF(mix2, 0));
  STOREU256_PS_HALF(result[2 * kernel_m], EXTRACT_PS_HALF(mix3, 0));
  STOREU256_PS_HALF(result[3 * kernel_m], EXTRACT_PS_HALF(mix4, 0));
  STOREU256_PS_HALF(result[4 * kernel_m], EXTRACT_PS_HALF(mix1, 1));
  STOREU256_PS_HALF(result[5 * kernel_m], EXTRACT_PS_HALF(mix2, 1));
  STOREU256_PS_HALF(result[6 * kernel_m], EXTRACT_PS_HALF(mix3, 1));
  STOREU256_PS_HALF(result[7 * kernel_m], EXTRACT_PS_HALF(mix4, 1));
#else
	STOREU_PS(result[0 * kernel_m], mix1);
	STOREU_PS(result[1 * kernel_m], mix2);
	STOREU_PS(result[2 * kernel_m], mix3);
	STOREU_PS(result[3 * kernel_m], mix4);
#endif
}

template<size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE FMAResult(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4,
                                                        float* result[], size_t length, size_t valid_lanes, size_t i_index, size_t j_index,
                                                        float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                                        bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                        float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  float bias1, bias2, bias3, bias4;
  if (bias != NULL) {
    bias1 = bias[i_index];
    bias2 = bias[i_index + 1];
    bias3 = bias[i_index + 2];
    bias4 = bias[i_index + 3];
  } else {
    bias1 = 0.0;
    bias2 = 0.0;
    bias3 = 0.0;
    bias4 = 0.0;
  }
#ifdef __AVX2__
  SIMDSITYPEHALF tmp1, tmp2, tmp3, tmp4;
  tmp1 = EXTRACT_SI128(sum1, 0); tmp2 = EXTRACT_SI128(sum2, 0);
  tmp3 = EXTRACT_SI128(sum3, 0); tmp4 = EXTRACT_SI128(sum4, 0);
  for (size_t ky = 0; ky < valid_lanes; ++ky) {
    if (length == 4) {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
      *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
      *(result[2 * kernel_n + ky]) = ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3;
      *(result[3 * kernel_n + ky]) = ratio_a[i_index + 3] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp4, 0) + kernel_sum[i_index + 3] * min_b[j_index + ky] + bias4;
    } else if (length == 3) {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
      *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
      *(result[2 * kernel_n + ky]) = ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3;
    } else if (length == 2) {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
      *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
    } else {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32_HALF(tmp1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
    }
    for (size_t l = 0; l < length; ++l) {
      if (conv_relu_fusion) {
        *(result[l * kernel_n + ky]) = fmaxf(*(result[l * kernel_n + ky]), 0.0f);
      } else if (conv_bn_fusion) {
        ScalarBN((*result[l * kernel_n + ky]), global_mean[i_index + l], mul_variance_coeff[i_index + l], (scale == NULL)? 1.0f : scale[i_index + l], (shift == NULL)? 0.0f : shift[i_index + l]);
      } else if (conv_bn_relu_fusion) {
        ScalarBN((*result[l * kernel_n + ky]), global_mean[i_index + l], mul_variance_coeff[i_index + l], (scale == NULL)? 1.0f : scale[i_index + l], (shift == NULL)? 0.0f : shift[i_index + l]);
        *(result[l * kernel_n + ky]) = fmaxf(*(result[l * kernel_n + ky]), 0.0f);
      } else if (conv_relu_bn_fusion) {
        *(result[l * kernel_n + ky]) = fmaxf(*(result[l * kernel_n + ky]), 0.0f);
        ScalarBN((*result[l * kernel_n + ky]), global_mean[i_index + l], mul_variance_coeff[i_index + l], (scale == NULL)? 1.0f : scale[i_index + l], (shift == NULL)? 0.0f : shift[i_index + l]);
      }
    }
    if (ky == 3) {
      tmp1 = EXTRACT_SI128(sum1, 1); tmp2 = EXTRACT_SI128(sum2, 1);
      tmp3 = EXTRACT_SI128(sum3, 1); tmp4 = EXTRACT_SI128(sum4, 1);
    } else {
      tmp1 = SRLI_SI128_HALF(tmp1, 4); tmp2 = SRLI_SI128_HALF(tmp2, 4);
      tmp3 = SRLI_SI128_HALF(tmp3, 4); tmp4 = SRLI_SI128_HALF(tmp4, 4);
    }
  }
#else
  for (size_t ky = 0; ky < valid_lanes; ++ky) {
    if (conv_relu_fusion) {
      if (length == 4) {
        *(result[0 * kernel_n + ky]) = fmaxf(ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1, 0.0f);
        *(result[1 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2, 0.0f);
        *(result[2 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3, 0.0f);
        *(result[3 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 3] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum4, 0) + kernel_sum[i_index + 3] * min_b[j_index + ky] + bias4, 0.0f);
      } else if (length == 3) {
        *(result[0 * kernel_n + ky]) = fmaxf(ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1, 0.0f);
        *(result[1 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2, 0.0f);
        *(result[2 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3, 0.0f);
      } else if (length == 2) {
        *(result[0 * kernel_n + ky]) = fmaxf(ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1, 0.0f);
        *(result[1 * kernel_n + ky]) = fmaxf(ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2, 0.0f);
      } else {
        *(result[0 * kernel_n + ky]) = fmaxf(ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1, 0.0f);
      }
    } else {
      if (length == 4) {
        *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
        *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
        *(result[2 * kernel_n + ky]) = ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3;
        *(result[3 * kernel_n + ky]) = ratio_a[i_index + 3] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum4, 0) + kernel_sum[i_index + 3] * min_b[j_index + ky] + bias4;
      } else if (length == 3) {
        *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
        *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
        *(result[2 * kernel_n + ky]) = ratio_a[i_index + 2] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum3, 0) + kernel_sum[i_index + 2] * min_b[j_index + ky] + bias3;
      } else if (length == 2) {
        *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
        *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum2, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + bias2;
      } else {
        *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum1, 0) + kernel_sum[i_index] * min_b[j_index + ky] + bias1;
      }
    }
		sum1 = SRLI_SI128(sum1, 4); sum2 = SRLI_SI128(sum2, 4);
		sum3 = SRLI_SI128(sum3, 4); sum4 = SRLI_SI128(sum4, 4);
  }
#endif
}


template <size_t kernel_k, typename kernel_function, typename sum_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes,
                                    kernel_function kernel, sum_function sum, reduce_function reduce, postprocess_function postprocess) {
  SIMDSITYPE ones = SET1_EPI16(1);
  SIMDSITYPE max_threshold = SET1_EPI16((INT16_MAX * fault_tolerance));
  SIMDSITYPE min_threshold = SET1_EPI16((INT16_MIN * fault_tolerance));
  INIT(c11); INIT(c12);
  INIT(c21); INIT(c22);
  INIT(c31); INIT(c32);
  INIT(c41); INIT(c42);
  INIT(sum1); INIT(sum2);
  INIT(sum3); INIT(sum4);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42, sum1, sum2, sum3, sum4, max_threshold, ones, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
    k -= kernel_k;
  }
  reduce(c11, c12, c21, c22, c31, c32, c41, c42, sum1, sum2, sum3, sum4);
  postprocess(sum1, sum2, sum3, sum4, result, length, valid_lanes);
}

template <size_t kernel_k, typename kernel_function, typename sum_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    kernel_function kernel, sum_function sum, reduce_function reduce, postprocess_function postprocess) {
  SIMDSITYPE ones = SET1_EPI16(1);
  SIMDPSTYPE zero = ZERO_PS();
  SIMDSITYPE max_threshold = SET1_EPI16((INT16_MAX * fault_tolerance));
  SIMDSITYPE min_threshold = SET1_EPI16((INT16_MIN * fault_tolerance));
  INIT(c11); INIT(c12);
  INIT(c21); INIT(c22);
  INIT(c31); INIT(c32);
  INIT(c41); INIT(c42);
  INIT(sum1); INIT(sum2);
  INIT(sum3); INIT(sum4);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42, sum1, sum2, sum3, sum4, max_threshold, ones, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c11, c12, c21, c22, c31, c32, c41, c42);
    k -= kernel_k;
  }
  reduce(c11, c12, c21, c22, c31, c32, c41, c42, sum1, sum2, sum3, sum4);
  postprocess(sum1, sum2, sum3, sum4, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift);
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes) {
#ifdef __AVX2__
  assert((kernel_m == 4) && (kernel_n == 8) && (kernel_k == 8));
  if ((length >= kernel_m) && (valid_lanes >= kernel_n)) {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, CommitBlockResult);
  } else {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, CommitResult);
  }
#else
  assert((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 8));
  if ((length >= kernel_m) && (valid_lanes >= kernel_n)) {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, CommitBlockResult);
  } else {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, CommitResult);
  }
#endif
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    bool is_block) {
#ifdef __AVX2__
  assert((kernel_m == 4) && (kernel_n == 8) && (kernel_k == 8));
  if (layout == NCHW) {
    if (is_block) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, NCHWFMABlockResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, FMAResult<kernel_m, kernel_n>);
    }
  } else {
    if (is_block) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, NHWCFMABlockResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, AVX2Kernel4x8x8, HaddPairReduce, PostHaddReduce, FMAResult<kernel_m, kernel_n>);
    }
  }
#else
  assert((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 8));
  if (layout == NCHW) {
    if (is_block) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, NCHWFMABlockResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, FMAResult<kernel_m, kernel_n>);
    }
  } else {
    if (is_block) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, NHWCFMABlockResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, SSE42Kernel4x4x8, HaddPairReduce, PostHaddReduce, FMAResult<kernel_m, kernel_n>);
    }
  }
#endif
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t feature_map_size_per_image, size_t feature_map_size_per_group, \
                                                                            size_t feature_map_size_per_channel) {
  size_t b0 = j_index / feature_map_size_per_channel;
  size_t b1 = (j_index + kernel_n) / feature_map_size_per_channel;
  if ((valid_m - i_index) >= kernel_m && (valid_n - j_index) >= kernel_n && (b0 == b1)) {
    NCHWGenrateBlockTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image, feature_map_size_per_group, feature_map_size_per_channel);
    return true;
  } else {
    NCHWGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image, feature_map_size_per_group, feature_map_size_per_channel);
    return false;
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t channel_per_group, size_t total_channels) {
  if ((valid_m - i_index) >= kernel_m && (valid_n - j_index) >= kernel_n) {
    NHWCGenrateBlockTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
    return true;
  } else {
    NHWCGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
    return false;
  }
}

} // namespace igemm4xn
} // namespace kernel
#endif // IGEMM4XN_X64
