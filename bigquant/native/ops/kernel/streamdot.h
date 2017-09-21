#include "../../base.h"
#include "../../common.h"

namespace kernel {
namespace dot {

static INLINE_SPECIFIER void INLINE_ATTRIBUTE StreamDotKernel(int8_t *&a, uint8_t *&b, SIMDSITYPE &c) {
  SIMDSITYPE k = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(a));
  SIMDSITYPE d = LOAD_SI(reinterpret_cast<SIMDSITYPE *>(b));
  c = ADDS_EPI16(MADD_EPI8(k, d), c);
  a += OPERAND_WIDTH;
  b += OPERAND_WIDTH;
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ReduceProcess(SIMDSITYPE &r, SIMDSITYPE &c) {
  const static SIMDSITYPE ones = SET1_EPI16(1);
  r = ADD_EPI16(MADD_EPI16(c, ones), r);
  c = ZEROS();
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t *&a, uint8_t *&b, SIMDSITYPE &c, SIMDSITYPE &r) {
#if UNROLL_NUM == 4
  StreamDotKernel(a, b, c);
  StreamDotKernel(a, b, c);
  StreamDotKernel(a, b, c);
  StreamDotKernel(a, b, c);
#else
  if (size_t i = 0; i < UNROLL_NUM; ++i) {
    StreamDotKernel(a, b, c);
  }
#endif
  ReduceProcess(r, c);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostProcess(int &result, SIMDSITYPE &r) {
#if defined(AVX512)
  const static uint16_t mask = 0xFFFF;
  result = MASK_REDUCEADD_EPI32(mask, r);
#elif defined(__AVX2__)
  SIMDSITYPE r1 = PERMUTE_SI128(r, r, 1);
  r = ADD_EPI32(r1, r);
  r = HADD_EPI32(r, r);
  r = HADD_EPI32(r, r);
  result = EXTRACT_EPI32_HALF(EXTRACT_SI128(r, 0), 0);
#else

#endif
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE PostProcess(float &result, SIMDSITYPE &r, float &ratio_a, float &a_sum,
                                                          float &ratio_b, float &min_b) {
#if defined(AVX512)
  int tmp_result;
  PostProcess(tmp_result, r);
  result = ratio_a * ratio_b * tmp_result + a_sum * min_b;
#elif defined(__AVX2__)
  SIMDSITYPE r1 = PERMUTE_SI128(r, r, 1);
  r = ADD_EPI32(r1, r);
  r = HADD_EPI32(r, r);
  r = HADD_EPI32(r, r);
  result = ratio_a * ratio_b * EXTRACT_EPI32_HALF(EXTRACT_SI128(r, 0), 0) + a_sum * min_b;
#else

#endif
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t *a, uint8_t *b, int &result, size_t length) {
  SIMDSITYPE r = ZEROS();
  SIMDSITYPE c = ZEROS();
  size_t k = length;
  while (k >= UNROLL_NUM * OPERAND_WIDTH) {
    KernelReduce(a, b, c, r);
    k -= UNROLL_NUM * OPERAND_WIDTH;
  }
  if (k >= OPERAND_WIDTH) {
    while (k >= OPERAND_WIDTH) {
      StreamDotKernel(a, b, c);
      k -= OPERAND_WIDTH;
    }
    ReduceProcess(r, c);
  }
  PostProcess(result, r);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t *a, uint8_t *b, float &result, size_t length,
                                                          float &ratio_a, float &a_sum, float &ratio_b, float &min_b) {
  SIMDSITYPE r = ZEROS();
  SIMDSITYPE c = ZEROS();
  size_t k = length;
  while (k >= UNROLL_NUM * OPERAND_WIDTH) {
    KernelReduce(a, b, c, r);
    k -= UNROLL_NUM * OPERAND_WIDTH;
  }
  if (k >= OPERAND_WIDTH) {
    while (k >= OPERAND_WIDTH) {
      StreamDotKernel(a, b, c);
      k -= OPERAND_WIDTH;
    }
    ReduceProcess(r, c);
  }
  PostProcess(result, r, ratio_a, a_sum, ratio_b, min_b);
}
}
}
