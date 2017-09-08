#ifndef OPS_QUANTIZE_H
#define OPS_QUANTIZE_H


#include "../base.h"
#include "./find_extreme.h"

#if defined(AVX512)
INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel16Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale, const SIMDPSTYPE &bias) {
  const static SIMDSITYPE shuffle8mask = SET1_EPI32((12 << 24) + (8 << 16) +  (4 << 8) +  0);
  const static SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  SIMDPSTYPE data = LOADU_PS(src); // Load 64B of data
  SIMDSITYPE int_result = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data, scale, bias)), shuffle8mask)); // FMA then convert float to int
  SIMDSITYPEQUARTER result = EXTRACT_SI128(int_result, 0); // Get Low 16B
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER*>(dst), result); // store
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX512Kernel64Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale, const SIMDPSTYPE &bias) {
  // TODO(Not fully optimized version but should working)
  const static SIMDSITYPE shuffle8mask = SET1_EPI32((12 << 24) + (8 << 16) +  (4 << 8) +  0);
  const static SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  SIMDPSTYPE data1 = LOADU_PS(src); // Load 64B of data
  SIMDPSTYPE data2 = LOADU_PS(src + PS_OPERAND_WIDTH); // Load 64B of data
  SIMDPSTYPE data3 = LOADU_PS(src + 2 * PS_OPERAND_WIDTH); // Load 64B of data
  SIMDPSTYPE data4 = LOADU_PS(src + 3 * PS_OPERAND_WIDTH); // Load 64B of data
  SIMDSITYPE int_result1 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data1, scale, bias)), shuffle8mask)); // FMA then convert float to int
  SIMDSITYPE int_result2 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data2, scale, bias)), shuffle8mask)); // FMA then convert float to int
  SIMDSITYPE int_result3 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data3, scale, bias)), shuffle8mask)); // FMA then convert float to int
  SIMDSITYPE int_result4 = PERMUTEX_EPI32(PERMUTE_INDEX, SHUFFLE_EPI8(PSTOEPI32(FMA_PS(data4, scale, bias)), shuffle8mask)); // FMA then convert float to int
  SIMDSITYPEQUARTER result1 = EXTRACT_SI128(int_result1, 0); // Get Low 16B
  SIMDSITYPEQUARTER result2 = EXTRACT_SI128(int_result2, 0); // Get Low 16B
  SIMDSITYPEQUARTER result3 = EXTRACT_SI128(int_result3, 0); // Get Low 16B
  SIMDSITYPEQUARTER result4 = EXTRACT_SI128(int_result4, 0); // Get Low 16B
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER*>(dst), result1); // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER*>(dst + 1 * PS_OPERAND_WIDTH), result2); // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER*>(dst + 2 * PS_OPERAND_WIDTH), result3); // store
  STOREU_SI_QUARTER(reinterpret_cast<SIMDSITYPEQUARTER*>(dst + 3 * PS_OPERAND_WIDTH), result4); // store
}

#elif defined(__AVX2__)
INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX2Kernel32Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale, const SIMDPSTYPE &bias) {
  // function should be reentrant
  const static SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  const static SIMDSITYPE shuffle32mask = SET_EPI32(0, 0, 0, 0, 0, 0, 4, 0);
  SIMDSITYPE data1 = PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data2 = PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 8), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data3 = PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 16), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data4 = PERMUTE_EPI32(SHUFFLE_EPI8(PSTOEPI32(FMA_PS(LOADU256_PS(src + 24), scale, bias)), shuffle8mask), shuffle32mask);
  SIMDSITYPE data = PERMUTE_SI128(UNPACKLO_EPI64(data1, data2), UNPACKLO_EPI64(data3, data4), 2 << 4);
  STOREU_SI256(reinterpret_cast<SIMDSITYPE*>(dst), data);
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE AVX2Kernel8Quantize(uint8_t *dst, float *src, const SIMDPSTYPE &scale, const SIMDPSTYPE &bias) {
  const static SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  const static SIMDSITYPE shuffle32mask = SET_EPI32(0, 0, 0, 0, 0, 0, 4, 0);
  SIMDPSTYPE data = LOADU256_PS(src); // Load 32B of data
  SIMDSITYPE int_result = PSTOEPI32(FMA_PS(data, scale, bias)); // FMA then convert float to int
  SIMDSITYPE shuffle_result_per_lane = SHUFFLE_EPI8(int_result, shuffle8mask); // Get byte 0, 4, 8, 12
  SIMDSITYPE shuffle_result = PERMUTE_EPI32(shuffle_result_per_lane, shuffle32mask);
  SIMDSITYPEHALF result = EXTRACT_SI128(shuffle_result, 0); // Get Low
  STORELO_EPI64_HALF(reinterpret_cast<SIMDSITYPEHALF*>(dst), result); // store
}
#else
INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel8Quantize(uint8_t *dst, float *src, SIMDPSTYPE &scale, SIMDPSTYPE &bias) {
  const static SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
  SIMDPSTYPE data1 = LOADU_PS(src);
  SIMDPSTYPE data2 = LOADU_PS(src + 4);
  SIMDPSTYPE fma_data1 = FMA_PS(data1, scale, bias);
  SIMDPSTYPE fma_data2 = FMA_PS(data2, scale, bias);
  SIMDSITYPE int_fma_data1 = PSTOEPI32(fma_data1);
  SIMDSITYPE int_fma_data2 = PSTOEPI32(fma_data2);
  SIMDSITYPE int_fma_data1_shuffle = SHUFFLE_EPI8(int_fma_data1, shuffle8mask);
  SIMDSITYPE int_fma_data2_shuffle = SHUFFLE_EPI8(int_fma_data2, shuffle8mask);
  SIMDSITYPE result = UNPACKLO_EPI32(int_fma_data1_shuffle, int_fma_data2_shuffle);
  STORELO_EPI64(reinterpret_cast<SIMDSITYPE*>(dst), result);
}

INLINE_SPECIFIER void INLINE_ATTRIBUTE SSE42Kernel16Quantize(uint8_t *dst, float *src, SIMDPSTYPE &scale, SIMDPSTYPE &bias) {
  const static SIMDSITYPE shuffle8mask = SET_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);
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
  STOREU_SI(reinterpret_cast<SIMDSITYPE*>(dst), UNPACKLO_EPI64(result1, result2));
}
#endif

template <typename SrcType>
void PadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = (std::abs(max) > std::abs(min))? (threshold / std::abs(max)) : (threshold / std::abs(min));
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<int8_t>(std::round(src[i] * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

template <typename SrcType>
void PadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold) {
  FindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

#if defined(AVX512)
template <>
void PadQuantize<float>(uint8_t *dst, size_t length, size_t pad_length, float *src, float &min, float &max, float &ratio, float threshold) {
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
void PadQuantize<float>(uint8_t *dst, size_t length, size_t pad_length, float *src, float &min, float &max, float &ratio, float threshold) {
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
void ParallelPadQuantize(int8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold) {
  OMPFindMinMaxValue(src, length, min, max);
  ratio = (std::abs(max) > std::abs(min))? (threshold / std::abs(max)) : (threshold / std::abs(min));
#pragma omp parallel for
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<int8_t>(std::round(src[i] * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}

template <typename SrcType>
void ParallelPadQuantize(uint8_t *dst, size_t length, size_t pad_length, SrcType *src, SrcType &min, SrcType &max, SrcType &ratio, float threshold) {
  OMPFindMinMaxValue(src, length, min, max);
  ratio = threshold / (max - min);
#pragma omp parallel for
  for (size_t i = 0; i < length; ++i) {
    dst[i] = static_cast<uint8_t>(std::round((src[i] - min) * ratio));
  }
  memset(dst + length, 0, pad_length - length);
}


template <typename DType>
void PadQuantize2D(int8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max, DType *ratio, float sw_threshold) {
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
void PadQuantize2D(uint8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min, DType *max, DType *ratio, float sw_threshold) {
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
void QuantizeIm2col(DType* data,
            size_t channels,
            size_t height, size_t width,
            size_t kernel_h, size_t kernel_w,
            size_t pad_h, size_t pad_w,
            size_t stride_h, size_t stride_w,
            size_t dilation_h, size_t dilation_w,
            uint8_t *data_col,
            DType *min,
            DType *max,
            DType *ratio,
            float sw_threshold = 255.0f) {
  using namespace std;
  size_t output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  size_t output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  size_t patch_size = channels * kernel_h * kernel_w;
  if (layout == NCHW) {
    for (size_t output_rows = 0; output_rows < output_h; ++output_rows) { // total output rows
      for (size_t output_col = 0; output_col < output_w; ++output_col) { // total output cols
        size_t output_col_index = output_rows * output_w + output_col;
        uint8_t *addr = data_col + patch_size * output_col_index;
        DType local_min = FLT_MAX;
        DType local_max = -FLT_MAX;
        int input_row = -pad_h + output_rows * stride_h; // startline of input rows
        int input_col = -pad_w + output_col * stride_w; // startline of input cols
        for (size_t channel = 0; channel < channels; ++channel) { // total channel && real start of one patch
          for (size_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) { // total kernel height
            for (size_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) { // total kernel width
              int row_index = input_row + kernel_row;
              int col_index = input_col + kernel_col;
              if (is_a_ge_zero_and_a_lt_b(row_index, height) &&
                is_a_ge_zero_and_a_lt_b(col_index, width)) {
                int index = channel * height * width + row_index * width + col_index;
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
        } // finish of one patch process
        DType scale = sw_threshold / (local_max - local_min);
        min[output_col_index] = local_min;
        max[output_col_index] = local_max;
        ratio[output_col_index] = 1.0f / scale;
        uint8_t padfill = static_cast<uint8_t>(std::round((-local_min * scale)));
        size_t offset = 0;
        for (size_t channel = 0; channel < channels; ++channel) { // total channel && real start of one patch
          for (size_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) { // total kernel height
            for (size_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) { // total kernel width
              int row_index = input_row + kernel_row;
              int col_index = input_col + kernel_col;
              if (is_a_ge_zero_and_a_lt_b(row_index, height) &&
                is_a_ge_zero_and_a_lt_b(col_index, width)) {
                int index = channel * height * width + row_index * width + col_index;
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
    for (size_t output_rows = 0; output_rows < output_h; ++output_rows) { // total output rows
      for (size_t output_col = 0; output_col < output_w; ++output_col) { // total output cols
        size_t output_col_index = output_rows * output_w + output_col;
        uint8_t *addr = data_col + patch_size * output_col_index;
        DType local_min = FLT_MAX;
        DType local_max = -FLT_MAX;
        int input_row = -pad_h + output_rows * stride_h; // startline of input rows
        int input_col = -pad_w + output_col * stride_w; // startline of input cols
        for (size_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) { // total kernel height
          for (size_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) { // total kernel width
            for (size_t channel = 0; channel < channels; ++channel) { // total channel && real start of one patch
              int row_index = input_row + kernel_row;
              int col_index = input_col + kernel_col;
              if (is_a_ge_zero_and_a_lt_b(row_index, height) &&
                is_a_ge_zero_and_a_lt_b(col_index, width)) {
                int index = row_index * width * channels + col_index * channels + channel;
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
        } // finish of one patch process
        DType scale = sw_threshold / (local_max - local_min);
        min[output_col_index] = local_min;
        max[output_col_index] = local_max;
        ratio[output_col_index] = 1.0f / scale;
        uint8_t padfill = static_cast<uint8_t>(std::round(-local_min * scale));
        size_t offset = 0;
        for (size_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) { // total kernel height
          for (size_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) { // total kernel width
            for (size_t channel = 0; channel < channels; ++channel) { // total channel && real start of one patch
              int row_index = input_row + kernel_row;
              int col_index = input_col + kernel_col;
              if (is_a_ge_zero_and_a_lt_b(row_index, height) &&
                is_a_ge_zero_and_a_lt_b(col_index, width)) {
                int index = row_index * width * channels + col_index * channels + channel;
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
void QuantizeIm2colRef(DType* data,
            size_t batch_size,
            size_t channels, // Now No Group Support
            size_t height, size_t width,
            size_t kernel_h, size_t kernel_w,
            size_t pad_h, size_t pad_w,
            size_t stride_h, size_t stride_w,
            size_t dilation_h, size_t dilation_w,
            uint8_t* data_col,
            DType* min,
            DType* max,
            DType* ratio,
            float sw_threshold) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t patch_size = kernel_h * kernel_w * channels;
  size_t srcsize_per_batch = channels * height * width;
  size_t dstsize_per_batch = patch_size * output_h * output_w;
  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t src_index = srcsize_per_batch * batch;
    size_t dst_index = dstsize_per_batch * batch;
    QuantizeIm2col<DType, layout>(data + src_index, channels,
            height, width,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            data_col + dst_index,
            min + batch * output_h * output_w,
            max + batch * output_h * output_w,
            ratio + batch * output_h * output_w,
            sw_threshold);
  }
}


#endif
