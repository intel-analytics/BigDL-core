#ifndef OPS_SHUFFLE_KERNEL_SSE42_IGEMM2X2X16_H
#define OPS_SHUFFLE_KERNEL_SSE42_IGEMM2X2X16_H
#include "../../base.h"

#if !defined(__AVX2__) && defined(__SSE4_2__)
namespace kernel {
namespace sse42_igemm2x2x16 {

template <typename kernel_function, typename reduce_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22,
  SIMDSITYPE &sum,
  SIMDSITYPE &threshold, SIMDSITYPE &ones,
  kernel_function kernel, reduce_function reduce) {
#ifdef __x86_64__
  const static size_t unroll_num = UNROLL_NUM;
  __asm__ __volatile__ (
    "xor r15, r15\n"
    "1:"
    "movdqa xmm0, [%1]\n" // xmm0 b1
    "add r15, 1\n"
    "movdqa xmm1, xmm0\n" // xmm1 b1
    "pmaddubsw xmm0, [%0]\n" // a11
    "pmaddubsw xmm1, [%0 + 16]\n" // a21
    "movdqa xmm2, [%1 + 16]\n" // xmm2 b2
    "movdqa xmm3, xmm2\n" // xmm3 b2
    "pmaddubsw xmm2, [%0]\n" // a12
    "pmaddubsw xmm3, [%0 + 16]\n" // a22
    "add %0, 32\n"
    "add %1, 32\n"
    "prefetcht0 [%0 + 128]\n"
    "prefetcht0 [%1 + 128]\n"
    "paddsw %2, xmm0\n"
    "paddsw %3, xmm2\n"
    "paddsw %4, xmm1\n"
    "paddsw %5, xmm3\n"
    "cmp r15, %6\n"
    "jne 1b\n"
    :"+r"(pa),"+r"(pb),"+x"(c11),"+x"(c12),"+x"(c21),"+x"(c22):"r"(unroll_num):"cc","r12","r13","r14","r15",
      "xmm0","xmm1","xmm2","xmm3","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15"
  );
#else
  for (size_t i = 0; i < UNROLL_NUM; ++i) {
    kernel(pa, pb, c11, c12, c21, c22);
  }
#endif
  reduce(c11, c12, c21, c22, sum, threshold, ones);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel2x2x16(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c11, SIMDSITYPE &c12,
  SIMDSITYPE &c21, SIMDSITYPE &c22) {
  SIMDSITYPE a1 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pa));
  SIMDSITYPE a2 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pa + 16));
  SIMDSITYPE b1 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pb));
  SIMDSITYPE b2 = LOAD_SI128(reinterpret_cast<SIMDSITYPE*>(pb + 16));
  c11 = ADDS_EPI16(c11, MADD_EPI8(b1, a1));
  c12 = ADDS_EPI16(c12, MADD_EPI8(b2, a1));
  c21 = ADDS_EPI16(c21, MADD_EPI8(b1, a2));
  c22 = ADDS_EPI16(c22, MADD_EPI8(b2, a2));
  pa += 32;
  pb += 32;
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE Reduce(SIMDSITYPE &c11, SIMDSITYPE &c12, SIMDSITYPE &c21, SIMDSITYPE &c22, SIMDSITYPE &accumulator) {
  SIMDSITYPE c11_lo = EPI16TOEPI32(c11);
  SIMDSITYPE c12_lo = EPI16TOEPI32(c12);
  SIMDSITYPE c21_lo = EPI16TOEPI32(c21);
  SIMDSITYPE c22_lo = EPI16TOEPI32(c22);
  SIMDSITYPE c11_hi = EPI16TOEPI32(SHUFFLE_EPI32(c11, 0x4e));
  SIMDSITYPE c12_hi = EPI16TOEPI32(SHUFFLE_EPI32(c12, 0x4e));
  SIMDSITYPE c21_hi = EPI16TOEPI32(SHUFFLE_EPI32(c21, 0x4e));
  SIMDSITYPE c22_hi = EPI16TOEPI32(SHUFFLE_EPI32(c22, 0x4e));
  SIMDSITYPE c11_sum = ADD_EPI32(c11_lo, c11_hi);
  SIMDSITYPE c12_sum = ADD_EPI32(c12_lo, c12_hi);
  SIMDSITYPE c21_sum = ADD_EPI32(c21_lo, c21_hi);
  SIMDSITYPE c22_sum = ADD_EPI32(c22_lo, c22_hi);
  c11 = ZEROS();
  c12 = ZEROS();
  c21 = ZEROS();
  c22 = ZEROS();
  accumulator = ADD_EPI32(HADD_EPI32(HADD_EPI32(c11_sum, c12_sum), HADD_EPI32(c21_sum, c22_sum)), accumulator);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ReduceWrapper(SIMDSITYPE &c11, SIMDSITYPE &c12, SIMDSITYPE &c21, SIMDSITYPE &c22, SIMDSITYPE &accumulator,
                          SIMDSITYPE &threshold, SIMDSITYPE &ones) {
  SIMDSITYPE saturated1 = MAX_EPI16(ABS_EPI16(c11), ABS_EPI16(c12));
  SIMDSITYPE saturated2 = MAX_EPI16(ABS_EPI16(c21), ABS_EPI16(c21));
  SIMDSITYPE saturated = MAX_EPI16(saturated1, saturated2);
  SIMDSITYPE flag = CMP_EPI16(saturated, threshold);
#ifdef _MSC_VER
  if (TESTZ_SI(ones, flag)) {
    // do nothing
  } else {
    Reduce(c11, c12, c21, c22, accumulator);
  }
#else
  asm goto(
      "ptest %0, %1\n"
      "jnz %l2\n"
      ::"x"(ones), "x"(flag)::streamreducewrapperend);
  return;
streamreducewrapperend:
  Reduce(c11, c12, c21, c22, accumulator);
  return;
#endif
}

template <size_t kernel_k, typename kernel_function, typename sum_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes,
                                    kernel_function kernel, sum_function sum, reduce_function reduce, postprocess_function postprocess) {
  SIMDSITYPE ones = SET1_EPI16(-1);
  SIMDSITYPE max_threshold = SET1_EPI16((INT16_MAX * fault_tolerance));
  SIMDSITYPE min_threshold = SET1_EPI16((INT16_MIN * fault_tolerance));
  INIT(c11); INIT(c12);
  INIT(c21); INIT(c22);
  INIT(accumulator);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c11, c12, c21, c22, accumulator, max_threshold, ones, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c11, c12, c21, c22);
    k -= kernel_k;
  }
  reduce(c11, c12, c21, c22, accumulator);
  postprocess(accumulator, result, length, valid_lanes);
}



template <size_t kernel_k, typename kernel_function, typename sum_function, typename reduce_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    kernel_function kernel, sum_function sum, reduce_function reduce, postprocess_function postprocess) {
  SIMDSITYPE ones = SET1_EPI16(-1);
  SIMDPSTYPE zero = ZERO_PS();
  SIMDSITYPE max_threshold = SET1_EPI16((INT16_MAX * fault_tolerance));
  SIMDSITYPE min_threshold = SET1_EPI16((INT16_MIN * fault_tolerance));
  INIT(c11); INIT(c12);
  INIT(c21); INIT(c22);
  INIT(accumulator);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c11, c12, c21, c22, accumulator, max_threshold, ones, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c11, c12, c21, c22);
    k -= kernel_k;
  }
  reduce(c11, c12, c21, c22, accumulator);
  postprocess(accumulator, result, length, valid_lanes, i_index, j_index,
              ratio_a, ratio_b, min_b, kernel_sum, bias,
              conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion,
              global_mean, mul_variance_coeff, scale, shift);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitBlockResult(SIMDSITYPE &sum, void* result[], size_t length, size_t valid_lanes) {
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[0]), sum);
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[1]), SRLI_SI128(sum, 8));
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitResult(SIMDSITYPE& sum, void* result[], size_t length, size_t valid_lanes) {
  SIMDSITYPE sum_hi = SRLI_SI128(sum, 8);
  for (int ky = 0; ky < valid_lanes; ++ky) {
    if (length == 2) {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum, 0);
      *(reinterpret_cast<int*>(result[1]) + ky) = EXTRACT_EPI32(sum_hi, 0);
    } else {
      *(reinterpret_cast<int*>(result[0]) + ky) = EXTRACT_EPI32(sum, 0);
    }
		sum = SRLI_SI128(sum, 4); sum_hi = SRLI_SI128(sum_hi, 4);
  }
}

template<size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE FMAResult(SIMDSITYPE &sum, float* result[], size_t length, size_t valid_lanes, size_t i_index, size_t j_index,
                                                        float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                                        bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                        float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  SIMDSITYPE sum_hi = SRLI_SI128(sum, 8);
  for (size_t ky = 0; ky < valid_lanes; ++ky) {
    if (length == 2) {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum, 0) + kernel_sum[i_index] * min_b[j_index + ky] + ((bias == NULL)? 0.0f : bias[i_index]);
      *(result[1 * kernel_n + ky]) = ratio_a[i_index + 1] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum_hi, 0) + kernel_sum[i_index + 1] * min_b[j_index + ky] + ((bias == NULL)? 0.0f : bias[i_index + 1]);
    } else {
      *(result[0 * kernel_n + ky]) = ratio_a[i_index] * ratio_b[j_index + ky] * EXTRACT_EPI32(sum, 0) + kernel_sum[i_index] * min_b[j_index + ky] + ((bias == NULL)? 0.0f : bias[i_index]);
    }
		sum = SRLI_SI128(sum, 4); sum_hi = SRLI_SI128(sum_hi, 4);
  }
}


template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes) {
  assert((kernel_m == 2) && (kernel_n == 2) && (kernel_k == 16));
  if ((length >= kernel_m) && (valid_lanes >= kernel_n)) {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, SSE42Kernel2x2x16, ReduceWrapper, Reduce, CommitBlockResult);
  } else {
    ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, SSE42Kernel2x2x16, ReduceWrapper, Reduce, CommitResult);
  }
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    bool is_block) {
  assert((kernel_m == 2) && (kernel_n == 2) && (kernel_k == 16));
  ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, std::min(length, kernel_m), std::min(valid_lanes, kernel_n), i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift, SSE42Kernel2x2x16, ReduceWrapper, Reduce, FMAResult<kernel_m, kernel_n>);
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t channel_per_group, size_t total_channels) {
  NHWCGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  return false;
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t feature_map_size_per_image, size_t feature_map_size_per_group, \
                                                                            size_t feature_map_size_per_channel) {
  NCHWGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image, feature_map_size_per_group, feature_map_size_per_channel);
  return false;
}


}
}
#endif
#endif // IGEMM2X2_X64
