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

#ifndef OPS_SHUFFLE_KERNEL_AVX512_IGEMM_8X8X8_H
#define OPS_SHUFFLE_KERNEL_AVX512_IGEMM_8X8X8_H
#include "../../base.h"

#if defined(AVX512)
namespace kernel {
namespace avx512_igemm8x8x8 {

template <size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE KernelReduce(int8_t *&pa, uint8_t *&pb, SIMDSITYPE sum[], size_t length) {
  size_t num = length / (kernel_k * UNROLL_NUM);
  size_t remain = (length % (kernel_k * UNROLL_NUM)) / kernel_k;
  const SIMDSITYPE ones = SET1_EPI16(1);
  __asm__ __volatile__(
      "cmp $0, %10\n"
      "je 2f\n"

      "xor %%rbx, %%rbx\n"
      ".align 2\n"
      "1:"
      "vpxorq %%zmm0, %%zmm0, %%zmm0\n"
      "vmovdqa32 %%zmm0, %%zmm1\n"
      "vmovdqa32 %%zmm0, %%zmm2\n"
      "vmovdqa32 %%zmm0, %%zmm3\n"
      "vmovdqa32 %%zmm0, %%zmm4\n"
      "vmovdqa32 %%zmm0, %%zmm5\n"
      "vmovdqa32 %%zmm0, %%zmm6\n"
      "vmovdqa32 %%zmm0, %%zmm7\n"

      "vmovdqa32 (%1), %%zmm8\n"      // %%zmm0 b
      "vmovdqa32 64(%1), %%zmm9\n"    // %%zmm0 b
      "vmovdqa32 128(%1), %%zmm10\n"  // %%zmm0 b
      "vmovdqa32 192(%1), %%zmm11\n"  // %%zmm0 b
      "add $256, %1\n"

      // 1st round

      "vbroadcastsd 0(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm0, %%zmm0\n"

      "vbroadcastsd 8(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm1, %%zmm1\n"

      "vbroadcastsd 16(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm2, %%zmm2\n"

      "vbroadcastsd 24(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm3, %%zmm3\n"

      "vbroadcastsd 32(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm4, %%zmm4\n"

      "vbroadcastsd 40(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm5, %%zmm5\n"

      "vbroadcastsd 48(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm6, %%zmm6\n"

      "vbroadcastsd 56(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm7, %%zmm7\n"

      "add $64, %0\n"

      // 2nt round

      "vbroadcastsd 0(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm0, %%zmm0\n"

      "vbroadcastsd 8(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm1, %%zmm1\n"

      "vbroadcastsd 16(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm2, %%zmm2\n"

      "vbroadcastsd 24(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm3, %%zmm3\n"

      "vbroadcastsd 32(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm4, %%zmm4\n"

      "vbroadcastsd 40(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm5, %%zmm5\n"

      "vbroadcastsd 48(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm6, %%zmm6\n"

      "vbroadcastsd 56(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm9, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm7, %%zmm7\n"

      "add $64, %0\n"

      // 3rd round

      "vbroadcastsd 0(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm0, %%zmm0\n"

      "vbroadcastsd 8(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm1, %%zmm1\n"

      "vbroadcastsd 16(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm2, %%zmm2\n"

      "vbroadcastsd 24(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm3, %%zmm3\n"

      "vbroadcastsd 32(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm4, %%zmm4\n"

      "vbroadcastsd 40(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm5, %%zmm5\n"

      "vbroadcastsd 48(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm6, %%zmm6\n"

      "vbroadcastsd 56(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm10, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm7, %%zmm7\n"

      "add $64, %0\n"

      // 4th round
      "vbroadcastsd 0(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm0, %%zmm0\n"

      "vbroadcastsd 8(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm1, %%zmm1\n"

      "vbroadcastsd 16(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm2, %%zmm2\n"

      "vbroadcastsd 24(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm3, %%zmm3\n"

      "vbroadcastsd 32(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm4, %%zmm4\n"

      "vbroadcastsd 40(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm5, %%zmm5\n"

      "vbroadcastsd 48(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm6, %%zmm6\n"

      "vbroadcastsd 56(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm11, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm7, %%zmm7\n"

      "add $64, %0\n"

      "vpmaddwd %12, %%zmm0, %%zmm0\n"
      "vpmaddwd %12, %%zmm1, %%zmm1\n"
      "vpmaddwd %12, %%zmm2, %%zmm2\n"
      "vpmaddwd %12, %%zmm3, %%zmm3\n"
      "vpmaddwd %12, %%zmm4, %%zmm4\n"
      "vpmaddwd %12, %%zmm5, %%zmm5\n"
      "vpmaddwd %12, %%zmm6, %%zmm6\n"
      "vpmaddwd %12, %%zmm7, %%zmm7\n"

      "vpaddd %%zmm0, %2, %2\n"
      "vpaddd %%zmm1, %3, %3\n"
      "vpaddd %%zmm2, %4, %4\n"
      "vpaddd %%zmm3, %5, %5\n"
      "vpaddd %%zmm4, %6, %6\n"
      "vpaddd %%zmm5, %7, %7\n"
      "vpaddd %%zmm6, %8, %8\n"
      "vpaddd %%zmm7, %9, %9\n"

      "add $1, %%rbx\n"
      "cmp %10, %%rbx\n"
      "jne 1b\n"

      "cmp $0, %11\n"
      "je 4f\n"

      "2:"
      "xor %%rbx, %%rbx\n"
      "vpxorq %%zmm0, %%zmm0, %%zmm0\n"
      "vmovdqa32 %%zmm0, %%zmm1\n"
      "vmovdqa32 %%zmm0, %%zmm2\n"
      "vmovdqa32 %%zmm0, %%zmm3\n"
      "vmovdqa32 %%zmm0, %%zmm4\n"
      "vmovdqa32 %%zmm0, %%zmm5\n"
      "vmovdqa32 %%zmm0, %%zmm6\n"
      "vmovdqa32 %%zmm0, %%zmm7\n"

      ".align 2\n"
      "3:"
      "vmovdqa32 (%1), %%zmm8\n"

      "vbroadcastsd 0(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm0, %%zmm0\n"

      "vbroadcastsd 8(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm1, %%zmm1\n"

      "vbroadcastsd 16(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm2, %%zmm2\n"

      "vbroadcastsd 24(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm3, %%zmm3\n"

      "vbroadcastsd 32(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm4, %%zmm4\n"

      "vbroadcastsd 40(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm5, %%zmm5\n"

      "vbroadcastsd 48(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm6, %%zmm6\n"

      "vbroadcastsd 56(%0), %%zmm12\n"
      "vpmaddubsw %%zmm12, %%zmm8, %%zmm13\n"
      "vpaddsw %%zmm13, %%zmm7, %%zmm7\n"

      "add $64, %0\n"
      "add $64, %1\n"
      "add $1, %%rbx\n"
      "cmp %11, %%rbx\n"
      "jne 3b\n"

      "vpmaddwd %12, %%zmm0, %%zmm0\n"
      "vpmaddwd %12, %%zmm1, %%zmm1\n"
      "vpmaddwd %12, %%zmm2, %%zmm2\n"
      "vpmaddwd %12, %%zmm3, %%zmm3\n"
      "vpmaddwd %12, %%zmm4, %%zmm4\n"
      "vpmaddwd %12, %%zmm5, %%zmm5\n"
      "vpmaddwd %12, %%zmm6, %%zmm6\n"
      "vpmaddwd %12, %%zmm7, %%zmm7\n"

      "vpaddd %%zmm0, %2, %2\n"
      "vpaddd %%zmm1, %3, %3\n"
      "vpaddd %%zmm2, %4, %4\n"
      "vpaddd %%zmm3, %5, %5\n"
      "vpaddd %%zmm4, %6, %6\n"
      "vpaddd %%zmm5, %7, %7\n"
      "vpaddd %%zmm6, %8, %8\n"
      "vpaddd %%zmm7, %9, %9\n"

      "4:"
      : "+r"(pa), "+r"(pb), "+v"(sum[0]), "+v"(sum[1]), "+v"(sum[2]), "+v"(sum[3]), "+v"(sum[4]), "+v"(sum[5]),
        "+v"(sum[6]), "+v"(sum[7])
      : "r"(num), "r"(remain), "x"(ones)
      : "cc", "rbx", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
        "zmm12", "zmm13");
}

template <size_t kernel_k, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance,
                                                          void *result[], size_t length, size_t valid_lanes,
                                                          postprocess_function postprocess) {
  SIMDSITYPE sum[8];
  for (size_t i = 0; i < 8; ++i) {
    sum[i] = ZEROS();
  }
  KernelReduce<kernel_k>(pa, pb, sum, k);
  postprocess(sum, result, length, valid_lanes);
}

template <size_t kernel_k, typename postprocess_function>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernel(
    int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance, float *result[], size_t length, size_t valid_lanes,
    size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum, float *bias,
    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean,
    float *mul_variance_coeff, float *scale, float *shift, postprocess_function postprocess) {
  SIMDSITYPE sum[8];
  for (size_t i = 0; i < 8; ++i) {
    sum[i] = ZEROS();
  }
  KernelReduce<kernel_k>(pa, pb, sum, k);
  postprocess(sum, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum, bias,
              conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
              mul_variance_coeff, scale, shift);
}

/*
static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitBlockResult(SIMDSITYPE &sum, void* result[], size_t length, size_t
valid_lanes) {
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[0]), sum);
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(result[1]), SRLI_SI128(sum, 8));
}
*/

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE CommitResult(SIMDSITYPE sum[], void *result[], size_t length,
                                                           size_t valid_lanes) {
  const static SIMDSITYPE permute_mask = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

  sum[0] = ADD_EPI32(BSRLI_EPI128(sum[0], 4), sum[0]);
  sum[1] = ADD_EPI32(BSRLI_EPI128(sum[1], 4), sum[1]);
  sum[2] = ADD_EPI32(BSRLI_EPI128(sum[2], 4), sum[2]);
  sum[3] = ADD_EPI32(BSRLI_EPI128(sum[3], 4), sum[3]);
  sum[4] = ADD_EPI32(BSRLI_EPI128(sum[4], 4), sum[4]);
  sum[5] = ADD_EPI32(BSRLI_EPI128(sum[5], 4), sum[5]);
  sum[6] = ADD_EPI32(BSRLI_EPI128(sum[6], 4), sum[6]);
  sum[7] = ADD_EPI32(BSRLI_EPI128(sum[7], 4), sum[7]);

  sum[0] = PERMUTEX_EPI32(permute_mask, sum[0]);
  sum[1] = PERMUTEX_EPI32(permute_mask, sum[1]);
  sum[2] = PERMUTEX_EPI32(permute_mask, sum[2]);
  sum[3] = PERMUTEX_EPI32(permute_mask, sum[3]);
  sum[4] = PERMUTEX_EPI32(permute_mask, sum[4]);
  sum[5] = PERMUTEX_EPI32(permute_mask, sum[5]);
  sum[6] = PERMUTEX_EPI32(permute_mask, sum[6]);
  sum[7] = PERMUTEX_EPI32(permute_mask, sum[7]);

  for (size_t m = 0; m < length; ++m) {
    for (size_t n = 0; n < valid_lanes; ++n) {
      int *tmp = reinterpret_cast<int *>(&sum[m]);
      *(reinterpret_cast<int *>(result[m]) + n) = tmp[n];
    }
  }
}

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NCHWBlockFMA(SIMDSITYPE sum[], float *result[], size_t length,
                                                           size_t valid_lanes, size_t i_index, size_t j_index,
                                                           float *ratio_a, float *ratio_b, float *min_b,
                                                           float *kernel_sum, float *bias, bool conv_relu_fusion,
                                                           bool conv_bn_fusion, bool conv_bn_relu_fusion,
                                                           bool conv_relu_bn_fusion, float *global_mean,
                                                           float *mul_variance_coeff, float *scale, float *shift) {
  const SIMDSITYPE permute_mask = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

  sum[0] = ADD_EPI32(BSRLI_EPI128(sum[0], 4), sum[0]);
  sum[1] = ADD_EPI32(BSRLI_EPI128(sum[1], 4), sum[1]);
  sum[2] = ADD_EPI32(BSRLI_EPI128(sum[2], 4), sum[2]);
  sum[3] = ADD_EPI32(BSRLI_EPI128(sum[3], 4), sum[3]);
  sum[4] = ADD_EPI32(BSRLI_EPI128(sum[4], 4), sum[4]);
  sum[5] = ADD_EPI32(BSRLI_EPI128(sum[5], 4), sum[5]);
  sum[6] = ADD_EPI32(BSRLI_EPI128(sum[6], 4), sum[6]);
  sum[7] = ADD_EPI32(BSRLI_EPI128(sum[7], 4), sum[7]);

  sum[0] = PERMUTEX_EPI32(permute_mask, sum[0]);
  sum[1] = PERMUTEX_EPI32(permute_mask, sum[1]);
  sum[2] = PERMUTEX_EPI32(permute_mask, sum[2]);
  sum[3] = PERMUTEX_EPI32(permute_mask, sum[3]);
  sum[4] = PERMUTEX_EPI32(permute_mask, sum[4]);
  sum[5] = PERMUTEX_EPI32(permute_mask, sum[5]);
  sum[6] = PERMUTEX_EPI32(permute_mask, sum[6]);
  sum[7] = PERMUTEX_EPI32(permute_mask, sum[7]);

  SIMDPSTYPEHALF simd_ratio_b = LOADU_PS_HALF(ratio_b + j_index);
  SIMDPSTYPEHALF simd_min_b = LOADU_PS_HALF(min_b + j_index);
  SIMDPSTYPEHALF simd_result[8];
  SIMDPSTYPEHALF simd_bias[8];
  SIMDPSTYPEHALF simd_coeffi[8];
  simd_coeffi[0] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index]), simd_ratio_b);
  simd_coeffi[1] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 1]), simd_ratio_b);
  simd_coeffi[2] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 2]), simd_ratio_b);
  simd_coeffi[3] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 3]), simd_ratio_b);
  simd_coeffi[4] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 4]), simd_ratio_b);
  simd_coeffi[5] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 5]), simd_ratio_b);
  simd_coeffi[6] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 6]), simd_ratio_b);
  simd_coeffi[7] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 7]), simd_ratio_b);
  if (bias != NULL) {
    simd_bias[0] = SET1_PS_HALF(bias[i_index]);
    simd_bias[1] = SET1_PS_HALF(bias[i_index + 1]);
    simd_bias[2] = SET1_PS_HALF(bias[i_index + 2]);
    simd_bias[3] = SET1_PS_HALF(bias[i_index + 3]);
    simd_bias[4] = SET1_PS_HALF(bias[i_index + 4]);
    simd_bias[5] = SET1_PS_HALF(bias[i_index + 5]);
    simd_bias[6] = SET1_PS_HALF(bias[i_index + 6]);
    simd_bias[7] = SET1_PS_HALF(bias[i_index + 7]);
  } else {
    simd_bias[0] = SET1_PS_HALF(0);
    simd_bias[1] = SET1_PS_HALF(0);
    simd_bias[2] = SET1_PS_HALF(0);
    simd_bias[3] = SET1_PS_HALF(0);
    simd_bias[4] = SET1_PS_HALF(0);
    simd_bias[5] = SET1_PS_HALF(0);
    simd_bias[6] = SET1_PS_HALF(0);
    simd_bias[7] = SET1_PS_HALF(0);
  }
  simd_result[0] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[0])), simd_coeffi[0],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index]), simd_bias[0]));
  simd_result[1] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[1])), simd_coeffi[1],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 1]), simd_bias[1]));
  simd_result[2] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[2])), simd_coeffi[2],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 2]), simd_bias[2]));
  simd_result[3] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[3])), simd_coeffi[3],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 3]), simd_bias[3]));
  simd_result[4] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[4])), simd_coeffi[4],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 4]), simd_bias[4]));
  simd_result[5] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[5])), simd_coeffi[5],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 5]), simd_bias[5]));
  simd_result[6] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[6])), simd_coeffi[6],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 6]), simd_bias[6]));
  simd_result[7] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[7])), simd_coeffi[7],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 7]), simd_bias[7]));

  STOREU_PS_HALF(result[0 * kernel_n], simd_result[0]);
  STOREU_PS_HALF(result[1 * kernel_n], simd_result[1]);
  STOREU_PS_HALF(result[2 * kernel_n], simd_result[2]);
  STOREU_PS_HALF(result[3 * kernel_n], simd_result[3]);
  STOREU_PS_HALF(result[4 * kernel_n], simd_result[4]);
  STOREU_PS_HALF(result[5 * kernel_n], simd_result[5]);
  STOREU_PS_HALF(result[6 * kernel_n], simd_result[6]);
  STOREU_PS_HALF(result[7 * kernel_n], simd_result[7]);
}

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NHWCBlockFMA(SIMDSITYPE sum[], float *result[], size_t length,
                                                           size_t valid_lanes, size_t i_index, size_t j_index,
                                                           float *ratio_a, float *ratio_b, float *min_b,
                                                           float *kernel_sum, float *bias, bool conv_relu_fusion,
                                                           bool conv_bn_fusion, bool conv_bn_relu_fusion,
                                                           bool conv_relu_bn_fusion, float *global_mean,
                                                           float *mul_variance_coeff, float *scale, float *shift) {
  const SIMDSITYPE permute_mask = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

  sum[0] = ADD_EPI32(BSRLI_EPI128(sum[0], 4), sum[0]);
  sum[1] = ADD_EPI32(BSRLI_EPI128(sum[1], 4), sum[1]);
  sum[2] = ADD_EPI32(BSRLI_EPI128(sum[2], 4), sum[2]);
  sum[3] = ADD_EPI32(BSRLI_EPI128(sum[3], 4), sum[3]);
  sum[4] = ADD_EPI32(BSRLI_EPI128(sum[4], 4), sum[4]);
  sum[5] = ADD_EPI32(BSRLI_EPI128(sum[5], 4), sum[5]);
  sum[6] = ADD_EPI32(BSRLI_EPI128(sum[6], 4), sum[6]);
  sum[7] = ADD_EPI32(BSRLI_EPI128(sum[7], 4), sum[7]);

  sum[0] = PERMUTEX_EPI32(permute_mask, sum[0]);
  sum[1] = PERMUTEX_EPI32(permute_mask, sum[1]);
  sum[2] = PERMUTEX_EPI32(permute_mask, sum[2]);
  sum[3] = PERMUTEX_EPI32(permute_mask, sum[3]);
  sum[4] = PERMUTEX_EPI32(permute_mask, sum[4]);
  sum[5] = PERMUTEX_EPI32(permute_mask, sum[5]);
  sum[6] = PERMUTEX_EPI32(permute_mask, sum[6]);
  sum[7] = PERMUTEX_EPI32(permute_mask, sum[7]);

  SIMDPSTYPEHALF simd_ratio_b = LOADU_PS_HALF(ratio_b + j_index);
  SIMDPSTYPEHALF simd_min_b = LOADU_PS_HALF(min_b + j_index);
  SIMDPSTYPEHALF simd_result[8];
  SIMDPSTYPEHALF simd_bias[8];
  SIMDPSTYPEHALF simd_coeffi[8];
  simd_coeffi[0] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index]), simd_ratio_b);
  simd_coeffi[1] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 1]), simd_ratio_b);
  simd_coeffi[2] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 2]), simd_ratio_b);
  simd_coeffi[3] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 3]), simd_ratio_b);
  simd_coeffi[4] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 4]), simd_ratio_b);
  simd_coeffi[5] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 5]), simd_ratio_b);
  simd_coeffi[6] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 6]), simd_ratio_b);
  simd_coeffi[7] = MUL_PS_HALF(SET1_PS_HALF(ratio_a[i_index + 7]), simd_ratio_b);
  if (bias != NULL) {
    simd_bias[0] = SET1_PS_HALF(bias[i_index]);
    simd_bias[1] = SET1_PS_HALF(bias[i_index + 1]);
    simd_bias[2] = SET1_PS_HALF(bias[i_index + 2]);
    simd_bias[3] = SET1_PS_HALF(bias[i_index + 3]);
    simd_bias[4] = SET1_PS_HALF(bias[i_index + 4]);
    simd_bias[5] = SET1_PS_HALF(bias[i_index + 5]);
    simd_bias[6] = SET1_PS_HALF(bias[i_index + 6]);
    simd_bias[7] = SET1_PS_HALF(bias[i_index + 7]);
  } else {
    simd_bias[0] = SET1_PS_HALF(0);
    simd_bias[1] = SET1_PS_HALF(0);
    simd_bias[2] = SET1_PS_HALF(0);
    simd_bias[3] = SET1_PS_HALF(0);
    simd_bias[4] = SET1_PS_HALF(0);
    simd_bias[5] = SET1_PS_HALF(0);
    simd_bias[6] = SET1_PS_HALF(0);
    simd_bias[7] = SET1_PS_HALF(0);
  }
  simd_result[0] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[0])), simd_coeffi[0],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index]), simd_bias[0]));
  simd_result[1] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[1])), simd_coeffi[1],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 1]), simd_bias[1]));
  simd_result[2] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[2])), simd_coeffi[2],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 2]), simd_bias[2]));
  simd_result[3] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[3])), simd_coeffi[3],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 3]), simd_bias[3]));
  simd_result[4] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[4])), simd_coeffi[4],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 4]), simd_bias[4]));
  simd_result[5] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[5])), simd_coeffi[5],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 5]), simd_bias[5]));
  simd_result[6] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[6])), simd_coeffi[6],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 6]), simd_bias[6]));
  simd_result[7] = FMA_PS_HALF(EPI32TOPS_HALF(CASTSI512TOSI256(sum[7])), simd_coeffi[7],
                               FMA_PS_HALF(simd_min_b, SET1_PS_HALF(kernel_sum[i_index + 7]), simd_bias[7]));
  // a1,a2,a3,a4,a5,a6,a7,a8
  // b1,b2,b3,b4,b5,b6,b7,b8
  // c1,c2,c3,c4,c5,c6,c7,c8
  // d1,d2,d3,d4,d5,d6,d7,d8
  // e1,e2,e3,e4,e5,e6,e7,e8
  // f1,f2,f3,f4,f5,f6,f7,f8
  // g1,g2,g3,g4,g5,g6,g7,g8
  // h1,h2,h3,h4,h5,h6,h7,h8

  // a1,b1,a2,b2,a5,b5,a6,b6
  // a3,b3,a4,b4,a7,b7,a8,b8
  // c1,d1,c2,d2,c5,d5,c6,d6
  // c3,d3,c4,d4,c7,d7,c8,d8
  // e1,f1,e2,f2,e5,f5,e6,f6
  // e3,f3,e4,f4,e7,f7,e8,f8
  // g1,h1,g2,h2,g5,h5,g6,h6
  // g3,h3,g4,h4,g7,h7,g8,h8

  SIMDPSTYPEHALF tmp1[8];
  tmp1[0] = UNPACKLO_PS_HALF(simd_result[0], simd_result[1]);
  tmp1[1] = UNPACKHI_PS_HALF(simd_result[0], simd_result[1]);
  tmp1[2] = UNPACKLO_PS_HALF(simd_result[2], simd_result[3]);
  tmp1[3] = UNPACKHI_PS_HALF(simd_result[2], simd_result[3]);
  tmp1[4] = UNPACKLO_PS_HALF(simd_result[4], simd_result[5]);
  tmp1[5] = UNPACKHI_PS_HALF(simd_result[4], simd_result[5]);
  tmp1[6] = UNPACKLO_PS_HALF(simd_result[6], simd_result[7]);
  tmp1[7] = UNPACKHI_PS_HALF(simd_result[6], simd_result[7]);

  // a1,b1,c1,d1,a5,b5,c5,d5
  // a2,b2,c2,d2,a6,b6,c6,d6
  // a3,b3,c3,d3,a7,b7,c7,d7
  // a4,b4,c4,d4,a8,b8,c8,d8
  // e1,f1,g1,h1,e5,f5,g5,h5
  // e2,f2,g2,h2,e6,f6,g6,h6
  // e3,f3,g3,h3,e7,f7,g7,h7
  // e4,f4,g4,h4,e8,f8,g8,h8

  simd_result[0] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKLO_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[0]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[2])));
  simd_result[1] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKHI_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[0]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[2])));
  simd_result[2] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKLO_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[1]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[3])));
  simd_result[3] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKHI_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[1]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[3])));
  simd_result[4] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKLO_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[4]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[6])));
  simd_result[5] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKHI_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[4]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[6])));
  simd_result[6] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKLO_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[5]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[7])));
  simd_result[7] = reinterpret_cast<SIMDPSTYPEHALF>(
      UNPACKHI_PD_HALF(reinterpret_cast<SIMDPDTYPEHALF>(tmp1[5]), reinterpret_cast<SIMDPDTYPEHALF>(tmp1[7])));

  // a1,b1,c1,d1,e1,f1,g1,h1
  // a2,b2,c2,d2,e2,f2,g2,h2
  // a3,b3,c3,d3,e3,f3,g3,h3
  // a4,b4,c4,d4,e4,f4,g4,h4
  // a5,b5,c5,d5,e5,f5,g5,h5
  // a6,b6,c6,d6,e6,f6,g6,h6
  // a7,b7,c7,d7,e7,f7,g7,h7
  // a8,b8,c8,d8,e8,f8,g8,h8

  tmp1[0] = PERMUTE2F128_PS_HALF(simd_result[0], simd_result[4], 0 + (2 << 4));
  tmp1[1] = PERMUTE2F128_PS_HALF(simd_result[1], simd_result[5], 0 + (2 << 4));
  tmp1[2] = PERMUTE2F128_PS_HALF(simd_result[2], simd_result[6], 0 + (2 << 4));
  tmp1[3] = PERMUTE2F128_PS_HALF(simd_result[3], simd_result[7], 0 + (2 << 4));
  tmp1[4] = PERMUTE2F128_PS_HALF(simd_result[0], simd_result[4], 1 + (3 << 4));
  tmp1[5] = PERMUTE2F128_PS_HALF(simd_result[1], simd_result[5], 1 + (3 << 4));
  tmp1[6] = PERMUTE2F128_PS_HALF(simd_result[2], simd_result[6], 1 + (3 << 4));
  tmp1[7] = PERMUTE2F128_PS_HALF(simd_result[3], simd_result[7], 1 + (3 << 4));

  STOREU_PS_HALF(result[0 * kernel_n], tmp1[0]);
  STOREU_PS_HALF(result[1 * kernel_n], tmp1[1]);
  STOREU_PS_HALF(result[2 * kernel_n], tmp1[2]);
  STOREU_PS_HALF(result[3 * kernel_n], tmp1[3]);
  STOREU_PS_HALF(result[4 * kernel_n], tmp1[4]);
  STOREU_PS_HALF(result[5 * kernel_n], tmp1[5]);
  STOREU_PS_HALF(result[6 * kernel_n], tmp1[6]);
  STOREU_PS_HALF(result[7 * kernel_n], tmp1[7]);
}

template <size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE FMAResult(SIMDSITYPE sum[], float *result[], size_t length,
                                                        size_t valid_lanes, size_t i_index, size_t j_index,
                                                        float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum,
                                                        float *bias, bool conv_relu_fusion, bool conv_bn_fusion,
                                                        bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
                                                        float *global_mean, float *mul_variance_coeff, float *scale,
                                                        float *shift) {
  const SIMDSITYPE permute_mask = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0);

  sum[0] = ADD_EPI32(BSRLI_EPI128(sum[0], 4), sum[0]);
  sum[1] = ADD_EPI32(BSRLI_EPI128(sum[1], 4), sum[1]);
  sum[2] = ADD_EPI32(BSRLI_EPI128(sum[2], 4), sum[2]);
  sum[3] = ADD_EPI32(BSRLI_EPI128(sum[3], 4), sum[3]);
  sum[4] = ADD_EPI32(BSRLI_EPI128(sum[4], 4), sum[4]);
  sum[5] = ADD_EPI32(BSRLI_EPI128(sum[5], 4), sum[5]);
  sum[6] = ADD_EPI32(BSRLI_EPI128(sum[6], 4), sum[6]);
  sum[7] = ADD_EPI32(BSRLI_EPI128(sum[7], 4), sum[7]);

  sum[0] = PERMUTEX_EPI32(permute_mask, sum[0]);
  sum[1] = PERMUTEX_EPI32(permute_mask, sum[1]);
  sum[2] = PERMUTEX_EPI32(permute_mask, sum[2]);
  sum[3] = PERMUTEX_EPI32(permute_mask, sum[3]);
  sum[4] = PERMUTEX_EPI32(permute_mask, sum[4]);
  sum[5] = PERMUTEX_EPI32(permute_mask, sum[5]);
  sum[6] = PERMUTEX_EPI32(permute_mask, sum[6]);
  sum[7] = PERMUTEX_EPI32(permute_mask, sum[7]);

  for (size_t m = 0; m < length; ++m) {
    for (size_t n = 0; n < valid_lanes; ++n) {
      int *tmp = reinterpret_cast<int *>(&sum[m]);
      *(reinterpret_cast<float *>(result[m * kernel_n + n])) = ratio_a[i_index + m] * ratio_b[j_index + n] * tmp[n] +
                                                               kernel_sum[i_index + m] * min_b[j_index + n] +
                                                               ((bias == NULL) ? 0.0f : bias[i_index + m]);
    }
  }
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(int8_t *&pa, uint8_t *&pb, size_t k,
                                                                 float fault_tolerance, void *result[], size_t length,
                                                                 size_t valid_lanes) {
  assert((kernel_m == 8) && (kernel_n == 8) && (kernel_k == 8));
  ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, CommitResult<kernel_m, kernel_n>);
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE ApplyKernelWrapper(
    int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance, float *result[], size_t length, size_t valid_lanes,
    size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum, float *bias,
    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean,
    float *mul_variance_coeff, float *scale, float *shift, bool is_block) {
  assert((kernel_m == 8) && (kernel_n == 8) && (kernel_k == 8));
  if (layout == NCHW) {
    if (is_block == false) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                            min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                            conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                            FMAResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                            min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                            conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                            NCHWBlockFMA<kernel_m, kernel_n>);
    }
  } else {
    if (is_block == false) {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                            min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                            conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                            FMAResult<kernel_m, kernel_n>);
    } else {
      ApplyKernel<kernel_k>(pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b,
                            min_b, kernel_sum, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion,
                            conv_relu_bn_fusion, global_mean, mul_variance_coeff, scale, shift,
                            NHWCBlockFMA<kernel_m, kernel_n>);
    }
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                      size_t valid_n, size_t i_index, size_t j_index,
                                                                      size_t cur_group, size_t channel_per_group,
                                                                      size_t total_channels) {
  if ((valid_m - i_index) >= kernel_m && (valid_n - j_index) >= kernel_n) {
    NHWCGenrateBlockTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                          channel_per_group, total_channels);
    return true;
  } else {
    NHWCGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                     channel_per_group, total_channels);
    return false;
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(
    DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, size_t cur_group,
    size_t feature_map_size_per_image, size_t feature_map_size_per_group, size_t feature_map_size_per_channel) {
  size_t b0 = j_index / feature_map_size_per_channel;
  size_t b1 = (j_index + kernel_n) / feature_map_size_per_channel;
  if ((valid_m - i_index) >= kernel_m && (valid_n - j_index) >= kernel_n && (b0 == b1)) {
    NCHWGenrateBlockTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                          feature_map_size_per_image, feature_map_size_per_group,
                                                          feature_map_size_per_channel);
    return true;
  } else {
    NCHWGenrateTargetAddr<float, kernel_m, kernel_n>(result, pc, valid_m, valid_n, i_index, j_index, cur_group,
                                                     feature_map_size_per_image, feature_map_size_per_group,
                                                     feature_map_size_per_channel);
    return false;
  }
}
}
}
#endif
#endif  // IGEMM2X2_X64
