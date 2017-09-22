#ifndef OPS_SHUFFLE_KERNEL_AVX512_IGEMM_4X4X64_H
#define OPS_SHUFFLE_KERNEL_AVX512_IGEMM_4X4X64_H
#include "../../base.h"

#if defined(AVX512)
namespace kernel {
namespace avx512_igemm4x4x64 {

template <typename kernel_function, typename reduce_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE
KernelReduce(int8_t *&pa, uint8_t *&pb, SIMDSITYPE &c00, SIMDSITYPE &c01, SIMDSITYPE &c02, SIMDSITYPE &c03,
             SIMDSITYPE &c10, SIMDSITYPE &c11, SIMDSITYPE &c12, SIMDSITYPE &c13, SIMDSITYPE &c20, SIMDSITYPE &c21,
             SIMDSITYPE &c22, SIMDSITYPE &c23, SIMDSITYPE &c30, SIMDSITYPE &c31, SIMDSITYPE &c32, SIMDSITYPE &c33,
             SIMDSITYPE &sum00, SIMDSITYPE &sum01, SIMDSITYPE &sum10, SIMDSITYPE &sum11, SIMDSITYPE &sum20,
             SIMDSITYPE &sum21, SIMDSITYPE &sum30, SIMDSITYPE &sum31, kernel_function kernel, reduce_function reduce) {
#if UNROLL_NUM == 4
  kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
  kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
  kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
  kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
#else
  for (size_t i = 0; i < UNROLL_NUM; ++i) {
    kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
  }
#endif
  reduce(c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, sum00, sum01, sum10, sum11,
         sum20, sum21, sum30, sum31);
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel4x4x64(int8_t *&pa, uint8_t *&pb, SIMDSITYPE &c00,
                                                                 SIMDSITYPE &c01, SIMDSITYPE &c02, SIMDSITYPE &c03,
                                                                 SIMDSITYPE &c10, SIMDSITYPE &c11, SIMDSITYPE &c12,
                                                                 SIMDSITYPE &c13, SIMDSITYPE &c20, SIMDSITYPE &c21,
                                                                 SIMDSITYPE &c22, SIMDSITYPE &c23, SIMDSITYPE &c30,
                                                                 SIMDSITYPE &c31, SIMDSITYPE &c32, SIMDSITYPE &c33) {
  /*
  SIMDSITYPE a0 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa));
  SIMDSITYPE a1 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + kernel_k));
  SIMDSITYPE a2 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + 2 * kernel_k));
  SIMDSITYPE a3 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + 3 * kernel_k));
  SIMDSITYPE b0 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pb));
  SIMDSITYPE b1 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pb + kernel_k));
  SIMDSITYPE b2 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pb + 2 * kernel_k));
  SIMDSITYPE b3 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pb + 3 * kernel_k));
  c00 = ADDS_EPI16(c00, MADD_EPI8(b0, a0)); c01 = ADDS_EPI16(c01, MADD_EPI8(b1, a0)); c02 = ADDS_EPI16(c02,
  MADD_EPI8(b2, a0)); c03 = ADDS_EPI16(c03, MADD_EPI8(b3, a0));
  c10 = ADDS_EPI16(c10, MADD_EPI8(b0, a1)); c11 = ADDS_EPI16(c11, MADD_EPI8(b1, a1)); c12 = ADDS_EPI16(c12,
  MADD_EPI8(b2, a1)); c13 = ADDS_EPI16(c13, MADD_EPI8(b3, a1));
  c20 = ADDS_EPI16(c20, MADD_EPI8(b0, a2)); c21 = ADDS_EPI16(c21, MADD_EPI8(b1, a2)); c22 = ADDS_EPI16(c22,
  MADD_EPI8(b2, a2)); c23 = ADDS_EPI16(c23, MADD_EPI8(b3, a2));
  c30 = ADDS_EPI16(c30, MADD_EPI8(b0, a3)); c31 = ADDS_EPI16(c31, MADD_EPI8(b1, a3)); c32 = ADDS_EPI16(c32,
  MADD_EPI8(b2, a3)); c33 = ADDS_EPI16(c33, MADD_EPI8(b3, a3));
  */
  SIMDSITYPE b0 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pb));
  SIMDSITYPE b1 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pb + kernel_k));
  SIMDSITYPE b2 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pb + 2 * kernel_k));
  SIMDSITYPE b3 = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pb + 3 * kernel_k));
  SIMDSITYPE a;
  a = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pa));
  c00 = ADDS_EPI16(c00, MADD_EPI8(b0, a));
  c01 = ADDS_EPI16(c01, MADD_EPI8(b1, a));
  c02 = ADDS_EPI16(c02, MADD_EPI8(b2, a));
  c03 = ADDS_EPI16(c03, MADD_EPI8(b3, a));
  a = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pa + kernel_k));
  c10 = ADDS_EPI16(c10, MADD_EPI8(b0, a));
  c11 = ADDS_EPI16(c11, MADD_EPI8(b1, a));
  c12 = ADDS_EPI16(c12, MADD_EPI8(b2, a));
  c13 = ADDS_EPI16(c13, MADD_EPI8(b3, a));
  a = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pa + 2 * kernel_k));
  c20 = ADDS_EPI16(c20, MADD_EPI8(b0, a));
  c21 = ADDS_EPI16(c21, MADD_EPI8(b1, a));
  c22 = ADDS_EPI16(c22, MADD_EPI8(b2, a));
  c23 = ADDS_EPI16(c23, MADD_EPI8(b3, a));
  a = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(pa + 3 * kernel_k));
  c30 = ADDS_EPI16(c30, MADD_EPI8(b0, a));
  c31 = ADDS_EPI16(c31, MADD_EPI8(b1, a));
  c32 = ADDS_EPI16(c32, MADD_EPI8(b2, a));
  c33 = ADDS_EPI16(c33, MADD_EPI8(b3, a));

  pa += kernel_m * kernel_k;
  pb += kernel_n * kernel_k;
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE Reduce(SIMDSITYPE &c00, SIMDSITYPE &c01, SIMDSITYPE &c02, SIMDSITYPE &c03,
                                                     SIMDSITYPE &c10, SIMDSITYPE &c11, SIMDSITYPE &c12, SIMDSITYPE &c13,
                                                     SIMDSITYPE &c20, SIMDSITYPE &c21, SIMDSITYPE &c22, SIMDSITYPE &c23,
                                                     SIMDSITYPE &c30, SIMDSITYPE &c31, SIMDSITYPE &c32, SIMDSITYPE &c33,
                                                     SIMDSITYPE &sum00, SIMDSITYPE &sum01, SIMDSITYPE &sum10,
                                                     SIMDSITYPE &sum11, SIMDSITYPE &sum20, SIMDSITYPE &sum21,
                                                     SIMDSITYPE &sum30, SIMDSITYPE &sum31) {
  const static SIMDSITYPE ones = SET1_EPI16(1);
  const static SIMDSITYPE low2high_mask = SET_EPI64(3, 2, 1, 0, 0, 0, 0, 0);
  const static SIMDSITYPE high2low_mask = SET_EPI64(0, 0, 0, 0, 7, 6, 5, 4);
  const static uint8_t blend_mask = (1 << 7) + (1 << 6) + (1 << 5) + (1 << 4);
  c00 = MADD_EPI16(c00, ones);
  c01 = MADD_EPI16(c01, ones);
  c02 = MADD_EPI16(c02, ones);
  c03 = MADD_EPI16(c03, ones);
  c10 = MADD_EPI16(c10, ones);
  c11 = MADD_EPI16(c11, ones);
  c12 = MADD_EPI16(c12, ones);
  c13 = MADD_EPI16(c13, ones);
  c20 = MADD_EPI16(c20, ones);
  c21 = MADD_EPI16(c21, ones);
  c22 = MADD_EPI16(c22, ones);
  c23 = MADD_EPI16(c23, ones);
  c30 = MADD_EPI16(c30, ones);
  c31 = MADD_EPI16(c31, ones);
  c32 = MADD_EPI16(c32, ones);
  c33 = MADD_EPI16(c33, ones);

  c00 = ADD_EPI32(c00, PERMUTEX_EPI64(high2low_mask, c00));
  c01 = ADD_EPI32(c01, PERMUTEX_EPI64(low2high_mask, c01));
  c02 = ADD_EPI32(c02, PERMUTEX_EPI64(high2low_mask, c02));
  c03 = ADD_EPI32(c03, PERMUTEX_EPI64(low2high_mask, c03));
  c10 = ADD_EPI32(c10, PERMUTEX_EPI64(high2low_mask, c10));
  c11 = ADD_EPI32(c11, PERMUTEX_EPI64(low2high_mask, c11));
  c12 = ADD_EPI32(c12, PERMUTEX_EPI64(high2low_mask, c12));
  c13 = ADD_EPI32(c13, PERMUTEX_EPI64(low2high_mask, c13));
  c20 = ADD_EPI32(c20, PERMUTEX_EPI64(high2low_mask, c20));
  c21 = ADD_EPI32(c21, PERMUTEX_EPI64(low2high_mask, c21));
  c22 = ADD_EPI32(c22, PERMUTEX_EPI64(high2low_mask, c22));
  c23 = ADD_EPI32(c23, PERMUTEX_EPI64(low2high_mask, c23));
  c30 = ADD_EPI32(c30, PERMUTEX_EPI64(high2low_mask, c30));
  c31 = ADD_EPI32(c31, PERMUTEX_EPI64(low2high_mask, c31));
  c32 = ADD_EPI32(c32, PERMUTEX_EPI64(high2low_mask, c32));
  c33 = ADD_EPI32(c33, PERMUTEX_EPI64(low2high_mask, c33));

  sum00 = ADD_EPI32(sum00, MASK_BLEND_EPI64(blend_mask, c00, c01));
  sum01 = ADD_EPI32(sum01, MASK_BLEND_EPI64(blend_mask, c02, c03));
  sum10 = ADD_EPI32(sum10, MASK_BLEND_EPI64(blend_mask, c10, c11));
  sum11 = ADD_EPI32(sum11, MASK_BLEND_EPI64(blend_mask, c12, c13));
  sum20 = ADD_EPI32(sum20, MASK_BLEND_EPI64(blend_mask, c20, c21));
  sum21 = ADD_EPI32(sum21, MASK_BLEND_EPI64(blend_mask, c22, c23));
  sum30 = ADD_EPI32(sum30, MASK_BLEND_EPI64(blend_mask, c30, c31));
  sum31 = ADD_EPI32(sum31, MASK_BLEND_EPI64(blend_mask, c32, c33));

  c00 = ZEROS();
  c01 = ZEROS();
  c02 = ZEROS();
  c03 = ZEROS();
  c10 = ZEROS();
  c11 = ZEROS();
  c12 = ZEROS();
  c13 = ZEROS();
  c20 = ZEROS();
  c21 = ZEROS();
  c22 = ZEROS();
  c23 = ZEROS();
  c30 = ZEROS();
  c31 = ZEROS();
  c32 = ZEROS();
  c33 = ZEROS();
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostReduce(SIMDSITYPE &accumulator, SIMDSITYPE &sum00, SIMDSITYPE &sum01,
                                                         SIMDSITYPE &sum10, SIMDSITYPE &sum11, SIMDSITYPE &sum20,
                                                         SIMDSITYPE &sum21, SIMDSITYPE &sum30, SIMDSITYPE &sum31) {
}
template <size_t kernel_k, typename kernel_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance,
                                                          void *result[], size_t length, size_t valid_lanes,
                                                          kernel_function kernel, reduce_function reduce,
                                                          postprocess_function postprocess) {
  INIT(c00);
  INIT(c01);
  INIT(c02);
  INIT(c03);
  INIT(c10);
  INIT(c11);
  INIT(c12);
  INIT(c13);
  INIT(c20);
  INIT(c21);
  INIT(c22);
  INIT(c23);
  INIT(c30);
  INIT(c31);
  INIT(c32);
  INIT(c33);
  INIT(sum00);
  INIT(sum01);
  INIT(sum10);
  INIT(sum11);
  INIT(sum20);
  INIT(sum21);
  INIT(sum30);
  INIT(sum31);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, sum00, sum01,
                 sum10, sum11, sum20, sum21, sum30, sum31, kernel, reduce);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
    k -= kernel_k;
  }
  reduce(c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, sum00, sum01, sum10, sum11,
         sum20, sum21, sum30, sum31);
  postprocess(sum00, sum01, sum10, sum11, sum20, sum21, sum30, sum31, result, length, valid_lanes);
}

template <size_t kernel_k, typename kernel_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE
ApplyKernel(int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance, float *result[], size_t length,
            size_t valid_lanes, size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b,
            float *kernel_sum, float *bias, bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion,
            bool conv_relu_bn_fusion, float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
            kernel_function kernel, reduce_function reduce, postprocess_function postprocess) {
  INIT(c00);
  INIT(c01);
  INIT(c02);
  INIT(c03);
  INIT(c10);
  INIT(c11);
  INIT(c12);
  INIT(c13);
  INIT(c20);
  INIT(c21);
  INIT(c22);
  INIT(c23);
  INIT(c30);
  INIT(c31);
  INIT(c32);
  INIT(c33);
  INIT(sum00);
  INIT(sum01);
  INIT(sum10);
  INIT(sum11);
  INIT(sum20);
  INIT(sum21);
  INIT(sum30);
  INIT(sum31);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, sum00, sum01,
                 sum10, sum11, sum20, sum21, sum30, sum31, kernel, reduce);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33);
    k -= kernel_k;
  }
  reduce(c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, sum00, sum01, sum10, sum11,
         sum20, sum21, sum30, sum31);
  postprocess(sum00, sum01, sum10, sum11, sum20, sum21, sum30, sum31, result, length, valid_lanes, i_index, j_index,
              ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
              conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift);
}
/*

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitBlockResult(SIMDSITYPE &sum, void* result[], size_t length, size_t
valid_lanes) {
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[0]), sum);
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[1]), SRLI_SI128(sum, 8));
}
*/

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitResult(SIMDSITYPE &sum00, SIMDSITYPE &sum01, SIMDSITYPE &sum10,
                                                           SIMDSITYPE &sum11, SIMDSITYPE &sum20, SIMDSITYPE &sum21,
                                                           SIMDSITYPE &sum30, SIMDSITYPE &sum31, void *result[],
                                                           size_t length, size_t valid_lanes) {
  const static uint16_t high_mask = 0xFF00;
  const static uint16_t low_mask = 0x00FF;
  SIMDSITYPE accumulator = SET_EPI32(MASK_REDUCEADD_EPI32(high_mask, sum31), MASK_REDUCEADD_EPI32(low_mask, sum31),
                                     MASK_REDUCEADD_EPI32(high_mask, sum30), MASK_REDUCEADD_EPI32(low_mask, sum30),
                                     MASK_REDUCEADD_EPI32(high_mask, sum21), MASK_REDUCEADD_EPI32(low_mask, sum21),
                                     MASK_REDUCEADD_EPI32(high_mask, sum20), MASK_REDUCEADD_EPI32(low_mask, sum20),
                                     MASK_REDUCEADD_EPI32(high_mask, sum11), MASK_REDUCEADD_EPI32(low_mask, sum11),
                                     MASK_REDUCEADD_EPI32(high_mask, sum10), MASK_REDUCEADD_EPI32(low_mask, sum10),
                                     MASK_REDUCEADD_EPI32(high_mask, sum01), MASK_REDUCEADD_EPI32(low_mask, sum01),
                                     MASK_REDUCEADD_EPI32(high_mask, sum00), MASK_REDUCEADD_EPI32(low_mask, sum00));
  int *tmp = reinterpret_cast<int *>(&accumulator);
  for (size_t m = 0; m < length; ++m) {
    for (size_t n = 0; n < valid_lanes; ++n) {
      *(reinterpret_cast<int *>(result[m]) + n) = tmp[m * kernel_n + n];
    }
  }
}

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE
FMAResult(SIMDSITYPE &sum00, SIMDSITYPE &sum01, SIMDSITYPE &sum10, SIMDSITYPE &sum11, SIMDSITYPE &sum20,
          SIMDSITYPE &sum21, SIMDSITYPE &sum30, SIMDSITYPE &sum31, float *result[], size_t length, size_t valid_lanes,
          size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum, float *bias,
          bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
          float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  const static uint16_t high_mask = 0xFF00;
  const static uint16_t low_mask = 0x00FF;
  SIMDSITYPE accumulator = SET_EPI32(MASK_REDUCEADD_EPI32(high_mask, sum31), MASK_REDUCEADD_EPI32(low_mask, sum31),
                                     MASK_REDUCEADD_EPI32(high_mask, sum30), MASK_REDUCEADD_EPI32(low_mask, sum30),
                                     MASK_REDUCEADD_EPI32(high_mask, sum21), MASK_REDUCEADD_EPI32(low_mask, sum21),
                                     MASK_REDUCEADD_EPI32(high_mask, sum20), MASK_REDUCEADD_EPI32(low_mask, sum20),
                                     MASK_REDUCEADD_EPI32(high_mask, sum11), MASK_REDUCEADD_EPI32(low_mask, sum11),
                                     MASK_REDUCEADD_EPI32(high_mask, sum10), MASK_REDUCEADD_EPI32(low_mask, sum10),
                                     MASK_REDUCEADD_EPI32(high_mask, sum01), MASK_REDUCEADD_EPI32(low_mask, sum01),
                                     MASK_REDUCEADD_EPI32(high_mask, sum00), MASK_REDUCEADD_EPI32(low_mask, sum00));
  int *tmp = reinterpret_cast<int *>(&accumulator);
  for (size_t m = 0; m < length; ++m) {
    for (size_t n = 0; n < valid_lanes; ++n) {
      *(reinterpret_cast<float *>(result[m * kernel_n + n])) =
          ratio_a[i_index + m] * ratio_b[j_index + n] * tmp[m * kernel_n + n] +
          kernel_sum[i_index + m] * min_b[j_index + n] + ((bias == NULL) ? 0.0f : bias[i_index]);
      /*
      std::cerr << ratio_a[i_index + m] << " " << ratio_b[j_index + n] << " " << tmp[m * kernel_n + n] << " " <<
      kernel_sum[i_index + m] << " " << min_b[j_index + n] << std::endl;
      std::cerr << *(reinterpret_cast<float*>(result[m * kernel_n + n])) << std::endl;
      */
    }
  }
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t *&pa, uint8_t *&pb, size_t k,
                                                                 float fault_tolerance, void *result[], size_t length,
                                                                 size_t valid_lanes) {
  assert((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 64));
  ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes,
                        AVX512Kernel4x4x64<kernel_m, kernel_n, kernel_k>, Reduce, CommitResult<kernel_m, kernel_n>);
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(
    int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance, float *result[], size_t length, size_t valid_lanes,
    size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum, float *bias,
    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean,
    float *mul_variance_coeff, float *scale, float *shift, bool is_block) {
  assert((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 64));
  if (layout == NCHW) {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                          min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                          conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                          AVX512Kernel4x4x64<kernel_m, kernel_n, kernel_k>, Reduce, FMAResult<kernel_m, kernel_n>);
  } else {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                          min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                          conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                          AVX512Kernel4x4x64<kernel_m, kernel_n, kernel_k>, Reduce, FMAResult<kernel_m, kernel_n>);
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                      size_t valid_n, size_t i_index, size_t j_index,
                                                                      size_t cur_group, size_t channel_per_group,
                                                                      size_t total_channels) {
  NHWCGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                   channel_per_group, total_channels);
  return false;
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(
    DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, size_t cur_group,
    size_t feature_map_size_per_image, size_t feature_map_size_per_group, size_t feature_map_size_per_channel) {
  NCHWGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                   feature_map_size_per_image, feature_map_size_per_group,
                                                   feature_map_size_per_channel);
  return false;
}
}
}
#endif
#endif  // IGEMM2X2_X64
