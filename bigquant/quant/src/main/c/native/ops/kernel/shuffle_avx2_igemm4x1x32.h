#ifndef IGEMM4X1XK_X64
#define IGEMM4X1XK_X64
#include "../../base.h"
namespace kernel {
namespace igemm4x1 {

template <typename kernel_function, typename reduce_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c1, SIMDSITYPE &c2,
  SIMDSITYPE &c3, SIMDSITYPE &c4,
  SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4,
  kernel_function kernel, reduce_function reduce) {
#if defined(__x86_64__) && UNROLL_NUM == 4
  __asm__ __volatile__ (
    "xor r15, r15\n"
    "1:"
    "vmovdqa ymm8, [%1]\n"
    "vmovdqa ymm0, [%0]\n"
    "vmovdqa ymm1, [%0 + 32]\n"
    "vmovdqa ymm2, [%0 + 64]\n"
    "vmovdqa ymm3, [%0 + 96]\n"
    "vpmaddubsw ymm0, ymm8, ymm0\n"
    "vpmaddubsw ymm1, ymm8, ymm1\n"
    "vpmaddubsw ymm2, ymm8, ymm2\n"
    "vpmaddubsw ymm3, ymm8, ymm3\n"
    "vpaddsw %2, %2, ymm0\n"
    "vpaddsw %3, %3, ymm1\n"
    "vpaddsw %4, %4, ymm2\n"
    "vpaddsw %5, %5, ymm3\n"
    "add r15, 1\n"
    "add %0, 128\n"
    "add %1, 32\n"
    "cmp r15, 4\n"
    "jne 1b\n"
    :"+r"(pa),"+r"(pb),"+x"(c1),"+x"(c2),"+x"(c3),"+x"(c4)::"cc","r12","r13","r14","r15",
      "ymm0","ymm1","ymm2","ymm3","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"
  );
#else
  for (size_t i = 0; i < UNROLL_NUM; ++i) {
    kernel(pa, pb, c1, c2, c3, c4);
  }
#endif
  reduce(c1, c2, c3, c4, sum1, sum2, sum3, sum4);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE Kernel4x1x32(int8_t* &pa, uint8_t* &pb,
  SIMDSITYPE &c1, SIMDSITYPE &c2,
  SIMDSITYPE &c3, SIMDSITYPE &c4) {
  SIMDSITYPE b = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pb));
  SIMDSITYPE a1 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa));
  c1 = ADDS_EPI16(c1, MADD_EPI8(b, a1));
  SIMDSITYPE a2 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + OPERAND_WIDTH * 1));
  c2 = ADDS_EPI16(c2, MADD_EPI8(b, a2));
  SIMDSITYPE a3 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + OPERAND_WIDTH * 2));
  c3 = ADDS_EPI16(c3, MADD_EPI8(b, a3));
  SIMDSITYPE a4 = LOAD_SI(reinterpret_cast<SIMDSITYPE*>(pa + OPERAND_WIDTH * 3));
  c4 = ADDS_EPI16(c4, MADD_EPI8(b, a4));
  pa += 4 * OPERAND_WIDTH;
  pb += OPERAND_WIDTH;
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE Reduce(SIMDSITYPE &c1, SIMDSITYPE &c2, SIMDSITYPE &c3, SIMDSITYPE &c4, \
                                                    SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4) {
  const static SIMDSITYPE one = SET1_EPI16(1);
  sum1 = ADD_EPI32(sum1, MADD_EPI16(c1, one));
  sum2 = ADD_EPI32(sum2, MADD_EPI16(c2, one));
  sum3 = ADD_EPI32(sum3, MADD_EPI16(c3, one));
  sum4 = ADD_EPI32(sum4, MADD_EPI16(c4, one));
  c1 = ZEROS();
  c2 = ZEROS();
  c3 = ZEROS();
  c4 = ZEROS();
}


static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostReduce(SIMDSITYPE &sum1, SIMDSITYPE &sum2, SIMDSITYPE &sum3, SIMDSITYPE &sum4, SIMDSITYPEHALF &accumulator) {
  SIMDSITYPE tmp1 = HADD_EPI32(sum1, sum2);
  SIMDSITYPE tmp2 = HADD_EPI32(sum3, sum4);
  SIMDSITYPE tmp = HADD_EPI32(tmp1, tmp2);
  accumulator = EXTRACT_SI128(ADD_EPI32(tmp, PERMUTE_SI128(tmp, tmp, 1)), 0);
}

template <size_t kernel_k, typename kernel_function, typename sum_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyStreamKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes,
                                    kernel_function kernel, sum_function sum, postprocess_function postprocess) {
  SIMDSITYPEHALF accumulator;
  INIT(c1); INIT(c2);
  INIT(c3); INIT(c4);
  INIT(sum1); INIT(sum2);
  INIT(sum3); INIT(sum4);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c1, c2, c3, c4, sum1, sum2, sum3, sum4, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c1, c2, c3, c4);
    k -= kernel_k;
  }
  sum(c1, c2, c3, c4, sum1, sum2, sum3, sum4);
  PostReduce(sum1, sum2, sum3, sum4, accumulator);
  postprocess(accumulator, result, length, valid_lanes);
}


template <size_t kernel_k, typename kernel_function, typename sum_function, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyStreamKernel(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    kernel_function kernel, sum_function sum, postprocess_function postprocess) {
  SIMDSITYPEHALF accumulator;
  INIT(c1); INIT(c2);
  INIT(c3); INIT(c4);
  INIT(sum1); INIT(sum2);
  INIT(sum3); INIT(sum4);
  while (k >= UNROLL_NUM * kernel_k) {
    KernelReduce(pa, pb, c1, c2, c3, c4, sum1, sum2, sum3, sum4, kernel, sum);
    k -= UNROLL_NUM * kernel_k;
  }
  while (k >= kernel_k) {
    kernel(pa, pb, c1, c2, c3, c4);
    k -= kernel_k;
  }
  sum(c1, c2, c3, c4, sum1, sum2, sum3, sum4);
  PostReduce(sum1, sum2, sum3, sum4, accumulator);
  postprocess(accumulator, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias,
    conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion,
    global_mean, mul_variance_coeff, scale, shift);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitResult(SIMDSITYPEHALF &accumulator, void* result[], size_t length, size_t valid_lanes) {
  if (length == 4) {
    *(reinterpret_cast<int*>(result[0])) = EXTRACT_EPI32_HALF(accumulator, 0);
    *(reinterpret_cast<int*>(result[1])) = EXTRACT_EPI32_HALF(accumulator, 1);
    *(reinterpret_cast<int*>(result[2])) = EXTRACT_EPI32_HALF(accumulator, 2);
    *(reinterpret_cast<int*>(result[3])) = EXTRACT_EPI32_HALF(accumulator, 3);
  } else if (length == 3) {
    *(reinterpret_cast<int*>(result[0])) = EXTRACT_EPI32_HALF(accumulator, 0);
    *(reinterpret_cast<int*>(result[1])) = EXTRACT_EPI32_HALF(accumulator, 1);
    *(reinterpret_cast<int*>(result[2])) = EXTRACT_EPI32_HALF(accumulator, 2);
  } else if (length == 2) {
    *(reinterpret_cast<int*>(result[0])) = EXTRACT_EPI32_HALF(accumulator, 0);
    *(reinterpret_cast<int*>(result[1])) = EXTRACT_EPI32_HALF(accumulator, 1);
  } else {
    *(reinterpret_cast<int*>(result[0])) = EXTRACT_EPI32_HALF(accumulator, 0);
  }
}

template<size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE StreamFMAResult(SIMDSITYPEHALF &accumulator, float* result[], size_t length, size_t valid_lanes, size_t i_index, size_t j_index,
                                                              float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                                              bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                              float *global_mean, float *mul_variance_coeff, float *scale, float *shift) {
  int* tmp = reinterpret_cast<int*>(&accumulator);
  for (size_t m = 0; m < length; ++m) {
    *(reinterpret_cast<float*>(result[m])) = ratio_a[i_index + m] * ratio_b[j_index] * tmp[m] + kernel_sum[i_index + m] * min_b[j_index] + bias[i_index + m];
  }

}


template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, void* result[], size_t length, size_t valid_lanes) {
  assert((kernel_m == 4) && (kernel_n == 1) && (kernel_k == 32));
  if ((length >= kernel_m) && (valid_lanes >= kernel_n)) {
    ApplyStreamKernel<kernel_k>(pa, pb, k, fault_tolerance, result, kernel_m, kernel_n, Kernel4x1x32, Reduce, CommitResult);
  } else {
    ApplyStreamKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, Kernel4x1x32, Reduce, CommitResult);
  }
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t* &pa, uint8_t* &pb, size_t k, float fault_tolerance, float* result[], size_t length, size_t valid_lanes,
                                    size_t i_index, size_t j_index, float* ratio_a, float* ratio_b, float *min_b, float *kernel_sum, float *bias,
                                    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                    float *global_mean, float *mul_variance_coeff, float *scale, float *shift,
                                    bool is_block) {
  assert((kernel_m == 4) && (kernel_n == 1) && (kernel_k == 32));
  ApplyStreamKernel<kernel_k>(pa, pb, k, fault_tolerance, result, std::min(length, kernel_m), std::min(valid_lanes, kernel_n), i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias,
                              conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion,
                              global_mean, mul_variance_coeff, scale, shift,
                              Kernel4x1x32, Reduce, StreamFMAResult<kernel_m, kernel_n>);
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t channel_per_group, size_t total_channels) {
  NHWCGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  return false;
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER int INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, \
                                                                            size_t cur_group, size_t feature_map_size_per_image, size_t feature_map_size_per_group, \
                                                                            size_t feature_map_size_per_channel) {
  NCHWGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image, feature_map_size_per_group, feature_map_size_per_channel);
  return false;
}

}
}

#endif //IGEMM4X1XK_X64
