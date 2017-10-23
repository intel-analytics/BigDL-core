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

#ifndef OPS_QUANTIZE_H
#define OPS_QUANTIZE_H

#include "../base.h"
#include "./find_extreme.h"

#if defined(AVX512)
INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel8Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale,
                                                             const SIMDPSTYPE &bias) {
  SIMDSITYPE shuffle8mask = SET1_EPI32((12 << 24) + (8 << 16) + (4 << 8) + 0);
  SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  SIMDPSTYPE data = _mm512_castps256_ps512(LOADU_PS_HALF(src));  // Load 64B of data
  SIMDSITYPE int_result = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data, scale, bias)),
                                                                     shuffle8mask));  // FMA then convert float to int
  SIMDSITYPEQUARTER result = EXTRACT_SI128(int_result, 0);                            // Get Low 16B
  STORELO_EPI64_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst), result);
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel16Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale,
                                                              const SIMDPSTYPE &bias) {
  SIMDSITYPE shuffle8mask = SET1_EPI32((12 << 24) + (8 << 16) + (4 << 8) + 0);
  SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  SIMDPSTYPE data = LOADU_PS(src);  // Load 64B of data
  SIMDSITYPE int_result = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data, scale, bias)),
                                                                     shuffle8mask));  // FMA then convert float to int
  SIMDSITYPEQUARTER result = EXTRACT_SI128(int_result, 0);                            // Get Low 16B
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst), result);              // store
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel64Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale,
                                                              const SIMDPSTYPE &bias) {
  // TODO(Not fully optimized version but should working)
  SIMDSITYPE shuffle8mask = SET1_EPI32((12 << 24) + (8 << 16) + (4 << 8) + 0);
  SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  SIMDPSTYPE data1 = LOADU_PS(src);                         // Load 64B of data
  SIMDPSTYPE data2 = LOADU_PS(src + PS_OPERAND_WIDTH);      // Load 64B of data
  SIMDPSTYPE data3 = LOADU_PS(src + 2 * PS_OPERAND_WIDTH);  // Load 64B of data
  SIMDPSTYPE data4 = LOADU_PS(src + 3 * PS_OPERAND_WIDTH);  // Load 64B of data
  SIMDSITYPE int_result1 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data1, scale, bias)),
                                                                      shuffle8mask));  // FMA then convert float to int
  SIMDSITYPE int_result2 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data2, scale, bias)),
                                                                      shuffle8mask));  // FMA then convert float to int
  SIMDSITYPE int_result3 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data3, scale, bias)),
                                                                      shuffle8mask));  // FMA then convert float to int
  SIMDSITYPE int_result4 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data4, scale, bias)),
                                                                      shuffle8mask));  // FMA then convert float to int
  SIMDSITYPEQUARTER result1 = EXTRACT_SI128(int_result1, 0);                           // Get Low 16B
  SIMDSITYPEQUARTER result2 = EXTRACT_SI128(int_result2, 0);                           // Get Low 16B
  SIMDSITYPEQUARTER result3 = EXTRACT_SI128(int_result3, 0);                           // Get Low 16B
  SIMDSITYPEQUARTER result4 = EXTRACT_SI128(int_result4, 0);                           // Get Low 16B
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst), result1);              // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst + 1 * PS_OPERAND_WIDTH), result2);  // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst + 2 * PS_OPERAND_WIDTH), result3);  // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER *>(dst + 3 * PS_OPERAND_WIDTH), result4);  // store
}

#elif defined(__AVX2__)
INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX2Kernel32Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale,
                                                            const SIMDPSTYPE &bias) {
  // function should be reentrant
  SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  SIMDSITYPE shuffle32mask = SET_EPI32(0, 0, 0, 0, 0, 0, 4, 0);
  SIMDSITYPE data1 =
      PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data2 =
      PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 8), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data3 =
      PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 16), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data4 =
      PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 24), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data = PERMUTE_SI128(UNPACKLO_EPI64(data1, data2), UNPACKLO_EPI64(data3, data4), 2 << 4);
  STOREU_SI256(reinterpret_cast<SIMDSITYPE *>(dst), data);
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX2Kernel8Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale,
                                                           const SIMDPSTYPE &bias) {
  SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  SIMDSITYPE shuffle32mask = SET_EPI32(0, 0, 0, 0, 0, 0, 4, 0);
  SIMDPSTYPE data = LOADU256_PS(src);                                           // Load 32B of data
  SIMDSITYPE int_result = PSTOEPI32(FMA_PS(data, scale, bias));                 // FMA then convert float to int
  SIMDSITYPE shuffle_result_per_lane = SHUFFLE_EPI8(int_result, shuffle8mask);  // Get byte 0, 4, 8, 12
  SIMDSITYPE shuffle_result = PERMUTE_EPI32(shuffle_result_per_lane, shuffle32mask);
  SIMDSITYPEHALF result = EXTRACT_SI128(shuffle_result, 0);             // Get Low
  STORELO_EPI64_HALF(reinterpret_cast<SIMDSITYPEHALF *>(dst), result);  // store
}
#else
INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel8Quantize(uint8_t *dst, float *src, SIMDPSTYPE &scale,
                                                            SIMDPSTYPE &bias) {
  SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  SIMDPSTYPE data1 = LOADU_PS(src);
  SIMDPSTYPE data2 = LOADU_PS(src + 4);
  SIMDPSTYPE fma_data1 = FMA_PS(data1, scale, bias);
  SIMDPSTYPE fma_data2 = FMA_PS(data2, scale, bias);
  SIMDSITYPE int_fma_data1 = PSTOEPI32(fma_data1);
  SIMDSITYPE int_fma_data2 = PSTOEPI32(fma_data2);
  SIMDSITYPE int_fma_data1_shuffle = SHUFFLE_EPI8(int_fma_data1, shuffle8mask);
  SIMDSITYPE int_fma_data2_shuffle = SHUFFLE_EPI8(int_fma_data2, shuffle8mask);
  SIMDSITYPE result = UNPACKLO_EPI32(int_fma_data1_shuffle, int_fma_data2_shuffle);
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE *>(dst), result);
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel16Quantize(uint8_t *dst, float *src, SIMDPSTYPE &scale,
                                                             SIMDPSTYPE &bias) {
  SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  SIMDPSTYPE data1 = LOADU_PS(src);
  SIMDPSTYPE data2 = LOADU_PS(src + 4);
  SIMDPSTYPE data3 = LOADU_PS(src + 8);
  SIMDPSTYPE data4 = LOADU_PS(src + 12);
  SIMDPSTYPE fma_data1 = FMA_PS(data1, scale, bias);
  SIMDPSTYPE fma_data2 = FMA_PS(data2, scale, bias);
  SIMDPSTYPE fma_data3 = FMA_PS(data3, scale, bias);
  SIMDPSTYPE fma_data4 = FMA_PS(data4, scale, bias);
  SIMDSITYPE int_fma_data1 = PSTOEPI32(fma_data1);
  SIMDSITYPE int_fma_data2 = PSTOEPI32(fma_data2);
  SIMDSITYPE int_fma_data3 = PSTOEPI32(fma_data3);
  SIMDSITYPE int_fma_data4 = PSTOEPI32(fma_data4);
  SIMDSITYPE int_fma_data1_shuffle = SHUFFLE_EPI8(int_fma_data1, shuffle8mask);
  SIMDSITYPE int_fma_data2_shuffle = SHUFFLE_EPI8(int_fma_data2, shuffle8mask);
  SIMDSITYPE int_fma_data3_shuffle = SHUFFLE_EPI8(int_fma_data3, shuffle8mask);
  SIMDSITYPE int_fma_data4_shuffle = SHUFFLE_EPI8(int_fma_data4, shuffle8mask);
  SIMDSITYPE result1 = UNPACKLO_EPI32(int_fma_data1_shuffle, int_fma_data2_shuffle);
  SIMDSITYPE result2 = UNPACKLO_EPI32(int_fma_data3_shuffle, int_fma_data4_shuffle);
  STOREU_SI(reinterpret_cast<SIMDSITYPE *>(dst), UNPACKLO_EPI64(result1, result2));
}
#endif

template <typename SrcType>
void PadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max,
                 SrcType &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = (std::abs(max) > std::abs(min)) ? (threshold / std::abs(max)) : (threshold / std::abs(min));
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<int8_t>(std::round(src[i] * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

template <typename SrcType>
void PadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max,
                 SrcType &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

#if defined(AVX512)
template <>
void PadQuantize<float>(uint8_t *dst, size_t length, size_t pad_length, float *src, float &min, float &max,
                        float &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
  SIMDPSTYPE simd_ratio = SET1_PS(ratio);
  SIMDPSTYPE simd_bias = SET1_PS(-ratio * min);
  for (size_t i = 0; i < length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i += PS_OPERAND_WIDTH) {
    AVX512Kernel16Quantize(dst + i, src + i, simd_ratio, simd_bias);
  }
  for (size_t i = length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}
#elif defined(__AVX2__)
template <>
void PadQuantize<float>(uint8_t *dst, size_t length, size_t pad_length, float *src, float &min, float &max,
                        float &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
  SIMDPSTYPE simd_ratio = SET1_PS(ratio);
  SIMDPSTYPE simd_bias = SET1_PS(-ratio * min);
  for (size_t i = 0; i < length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i += PS_OPERAND_WIDTH) {
    AVX2Kernel8Quantize(dst + i, src + i, simd_ratio, simd_bias);
  }
  for (size_t i = length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}
#endif

template <typename SrcType>
void ParallelPadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max,
                         SrcType &ratio, float threshold) {
  OMPFindMinMaxValue(src, length, min, max);
  ratio = (std::abs(max) > std::abs(min)) ? (threshold / std::abs(max)) : (threshold / std::abs(min));
#pragma omp parallel for
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<int8_t>(std::round(src[i] * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

template <typename SrcType>
void ParallelPadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max,
                         SrcType &ratio, float threshold) {
  OMPFindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
#pragma omp parallel for
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

template <typename DType>
void PadQuantize2D(int8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max,
                   DType *ratio, float sw_threshold) {
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t src_offset = i * n;
    size_t dst_offset = i * pad_n;
    if (i < m) {
      PadQuantize(dst + dst_offset, n, pad_n, src + src_offset, min[i], max[i], ratio[i], sw_threshold);
    } else {
      memset(dst + dst_offset, 0, pad_n);
    }
  }
}

template <typename DType>
void PadQuantize2D(uint8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max,
                   DType *ratio, float sw_threshold) {
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t src_offset = i * n;
    size_t dst_offset = i * pad_n;
    if (i < m) {
      PadQuantize(dst + dst_offset, n, pad_n, src + src_offset, min[i], max[i], ratio[i], sw_threshold);
    } else {
      memset(dst + dst_offset, 0, pad_n);
    }
  }
}

template <typename DType, LAYOUT layout>
void QuantizeIm2col(DType *data, size_t channels, size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                    size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                    uint8_t *data_col, DType *min, DType *max, DType *ratio, float sw_threshold = 255.0f) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t patch_size = channels * kernel_h * kernel_w;
  if (layout == NCHW) {
    for (size_t o_y = 0; o_y < output_h; ++o_y) {  // total output rows
      for (size_t o_x = 0; o_x < output_w; ++o_x) {   // total output cols
        size_t out_spatial_id = o_y * output_w + o_x;
        uint8_t *addr = data_col + patch_size * out_spatial_id;
        DType local_min = FLT_MAX;
        DType local_max = -FLT_MAX;
        int conv_window_y = -pad_h + o_y * stride_h;           // startline of input rows
        int conv_window_x = -pad_w + o_x * stride_w;            // startline of input cols
        for (size_t channel = 0; channel < channels; ++channel) {  // total channel && real start of one patch
          for (size_t k_y = 0; k_y < kernel_h; k_y++) {    // total kernel height
            for (size_t k_x = 0; k_x < kernel_w; k_x++) {  // total kernel width
              int in_y = conv_window_y + k_y;
              int in_x = conv_window_x + k_x;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                int index = channel * height * width + in_y * width + in_x;
                DType value = data[index];
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              } else {
                DType value = 0;
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              }
            }
          }
        }  // finish of one patch process
        DType scale = sw_threshold / (local_max - local_min);
        min[out_spatial_id] = local_min;
        max[out_spatial_id] = local_max;
        ratio[out_spatial_id] = 1.0f / scale;
        uint8_t padfill = static_cast<uint8_t>(std::round((-local_min * scale)));
        size_t offset = 0;
        for (size_t channel = 0; channel < channels; ++channel) {  // total channel && real start of one patch
          for (size_t k_y = 0; k_y < kernel_h; k_y++) {    // total kernel height
            for (size_t k_x = 0; k_x < kernel_w; k_x++) {  // total kernel width
              int in_y = conv_window_y + k_y;
              int in_x = conv_window_x + k_x;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                int index = channel * height * width + in_y * width + in_x;
                *(addr + offset) = static_cast<uint8_t>(std::round((data[index] - local_min) * scale));
              } else {
                *(addr + offset) = padfill;
              }
              ++offset;
            }
          }
        }
      }
    }
  } else {
    for (size_t o_y = 0; o_y < output_h; ++o_y) {  // total output rows
      for (size_t o_x = 0; o_x < output_w; ++o_x) {   // total output cols
        size_t out_spatial_id = o_y * output_w + o_x;
        uint8_t *addr = data_col + patch_size * out_spatial_id;
        DType local_min = FLT_MAX;
        DType local_max = -FLT_MAX;
        int conv_window_y = -pad_h + o_y * stride_h;                      // startline of input rows
        int conv_window_x = -pad_w + o_x * stride_w;                       // startline of input cols
        for (size_t k_y = 0; k_y < kernel_h; k_y++) {    // total kernel height
          for (size_t k_x = 0; k_x < kernel_w; k_x++) {  // total kernel width
            for (size_t channel = 0; channel < channels; ++channel) {  // total channel && real start of one patch
              int in_y = conv_window_y + k_y;
              int in_x = conv_window_x + k_x;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                int index = in_y * width * channels + in_x * channels + channel;
                DType value = data[index];
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              } else {
                DType value = 0;
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              }
            }
          }
        }  // finish of one patch process
        DType scale = sw_threshold / (local_max - local_min);
        min[out_spatial_id] = local_min;
        max[out_spatial_id] = local_max;
        ratio[out_spatial_id] = 1.0f / scale;
        uint8_t padfill = static_cast<uint8_t>(std::round(-local_min * scale));
        size_t offset = 0;
        for (size_t k_y = 0; k_y < kernel_h; k_y++) {    // total kernel height
          for (size_t k_x = 0; k_x < kernel_w; k_x++) {  // total kernel width
            for (size_t channel = 0; channel < channels; ++channel) {  // total channel && real start of one patch
              int in_y = conv_window_y + k_y;
              int in_x = conv_window_x + k_x;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                int index = in_y * width * channels + in_x * channels + channel;
                *(addr + offset) = static_cast<uint8_t>(std::round((data[index] - local_min) * scale));
              } else {
                *(addr + offset) = padfill;
              }
              ++offset;
            }
          }
        }
      }
    }
  }
}

template <typename DType, LAYOUT layout>
void QuantizeIm2colRef(DType *data, size_t batch_size,
                       size_t channels,  // Now No Group Support
                       size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h, size_t pad_w,
                       size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                       uint8_t *data_col, DType *min, DType *max, DType *ratio,
                       float sw_threshold) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t patch_size = kernel_h * kernel_w * channels;
  size_t srcsize_per_batch = channels * height * width;
  size_t dstsize_per_batch = patch_size * output_h * output_w;
  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t src_index = srcsize_per_batch * batch;
    size_t dst_index = dstsize_per_batch * batch;
    QuantizeIm2col<DType, layout>(data + src_index, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                                  stride_w, dilation_h, dilation_w, data_col + dst_index,
                                  min + batch * output_h * output_w, max + batch * output_h * output_w,
                                  ratio + batch * output_h * output_w, sw_threshold);
  }
}

#endif
