#ifndef PAD_SHUFFLE_H
#define PAD_SHUFFLE_H

#include "../../base.h"
namespace shuffle {
template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadShuffle2D(DType *dst, size_t m, size_t n, DType *src) {
  size_t pad_m = GetAlignmentLength(m, shuffle_rows);
  size_t pad_n = GetAlignmentLength(n, shuffle_cols);
  size_t shuffle_cols_num = n / shuffle_cols * shuffle_cols;
  size_t patch_size = shuffle_cols * shuffle_rows;
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t x_block_id = i / shuffle_rows;
    size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
    size_t dst_index = x_block_id * shuffle_rows * pad_n + offset_in_block;
    size_t src_index = i * n;
    bool iltm = (i < m);
    size_t j;
    if (iltm) {  // if i < m
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        memcpy(&dst[dst_index], &src[src_index], shuffle_cols);
        dst_index += patch_size;
        src_index += shuffle_cols;
      }
    } else {  // if i >= m
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        memset(&dst[dst_index], 0, shuffle_cols);
        dst_index += patch_size;
        src_index += shuffle_cols;
      }
    }
    // boundary
    for (j = shuffle_cols_num; j < n; ++j) {
      // if i < m && j < n, assignment ans shift the index else just return zero
      dst[dst_index++] = iltm ? src[src_index++] : 0;
    }
    for (j = n; j < pad_n; ++j) {
      dst[dst_index++] = 0;
    }
  }
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle(int8_t *dst, size_t m, size_t n, DType *src, DType &min, DType &max, DType &ratio,
                        float sw_threshold) {
  size_t pad_m = GetAlignmentLength(m, shuffle_rows);
  size_t pad_n = GetAlignmentLength(n, shuffle_cols);
  size_t shuffle_cols_num = n / shuffle_cols * shuffle_cols;
  size_t patch_size = shuffle_cols * shuffle_rows;
  OMPFindMinMaxValue(src, m * n, min, max);
  float scale = std::abs(((max + min) > 0) ? (1.0 * sw_threshold / max) : (1.0 * sw_threshold / min));
  ratio = 1.0 / scale;
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t x_block_id = i / shuffle_rows;
    size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
    size_t dst_index = x_block_id * shuffle_rows * pad_n + offset_in_block;
    size_t src_index = i * n;
    bool iltm = (i < m);
    size_t j;
    if (iltm) {  // i lt m; FindMinMaxValue; GetRatio and Quantize
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        for (size_t k = 0; k < shuffle_cols; ++k) {
          dst[dst_index + k] = static_cast<int8_t>(std::round(src[src_index + k] * scale));
        }
        dst_index += patch_size;
        src_index += shuffle_cols;
      }
      for (j = shuffle_cols_num; j < n; ++j) {
        dst[dst_index++] = static_cast<int8_t>(std::round(src[src_index++] * scale));
      }
      memset(&dst[dst_index], 0, pad_n - n);
    } else {  // i >= m; memset;
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        memset(&dst[dst_index], 0, shuffle_cols);
        dst_index += patch_size;
      }
      memset(&dst[dst_index], 0, pad_n - shuffle_cols_num);
    }
  }
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle2D(uint8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min,
                          DType *max, DType *ratio, float sw_threshold) {
  assert(GetAlignmentLength(m, shuffle_rows) == pad_m);
  assert(GetAlignmentLength(n, shuffle_cols) == pad_n);
  size_t shuffle_cols_num = n / shuffle_cols * shuffle_cols;
  size_t patch_size = shuffle_cols * shuffle_rows;
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t x_block_id = i / shuffle_rows;
    size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
    size_t dst_index = x_block_id * shuffle_rows * pad_n + offset_in_block;
    size_t src_index = i * n;
    bool iltm = (i < m);
    size_t j;
    if (iltm) {  // i lt m; FindMinMaxValue; GetRatio and Quantize
      FindMinMaxValue(src + src_index, n, min[i], max[i]);
      DType scale = sw_threshold / (max[i] - min[i]);
      ratio[i] = 1.0 / scale;
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        for (size_t k = 0; k < shuffle_cols; ++k) {
          dst[dst_index + k] = static_cast<uint8_t>(std::round((src[src_index + k] - min[i]) * scale));
        }
        dst_index += patch_size;
        src_index += shuffle_cols;
      }
      for (j = shuffle_cols_num; j < n; ++j) {
        dst[dst_index++] = static_cast<uint8_t>(std::round((src[src_index++] - min[i]) * scale));
      }
      memset(&dst[dst_index], 0, pad_n - n);
    } else {  // i >= m; memset;
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        memset(&dst[dst_index], 0, shuffle_cols);
        dst_index += patch_size;
      }
      memset(&dst[dst_index], 0, pad_n - shuffle_cols_num);
    }
  }
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffle2D(int8_t *dst, size_t m, size_t n, size_t pad_m, size_t pad_n, DType *src, DType *min,
                          DType *max, DType *ratio, float sw_threshold) {
  assert(GetAlignmentLength(m, shuffle_rows) == pad_m);
  assert(GetAlignmentLength(n, shuffle_cols) == pad_n);
  size_t shuffle_cols_num = n / shuffle_cols * shuffle_cols;
  size_t patch_size = shuffle_cols * shuffle_rows;
#pragma omp parallel for proc_bind(close)
  for (size_t i = 0; i < pad_m; ++i) {
    size_t x_block_id = i / shuffle_rows;
    size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
    size_t dst_index = x_block_id * shuffle_rows * pad_n + offset_in_block;
    size_t src_index = i * n;
    bool iltm = (i < m);
    size_t j;
    if (iltm) {  // i lt m; FindMinMaxValue; GetRatio and Quantize
      FindMinMaxValue(src + src_index, n, min[i], max[i]);
      DType scale =
          std::abs(max[i]) > std::abs(min[i]) ? (sw_threshold / std::abs(max[i])) : (sw_threshold / std::abs(min[i]));
      ratio[i] = 1.0 / scale;
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        for (size_t k = 0; k < shuffle_cols; ++k) {
          dst[dst_index + k] = static_cast<int8_t>(std::round(src[src_index + k] * scale));
        }
        dst_index += patch_size;
        src_index += shuffle_cols;
      }
      for (j = shuffle_cols_num; j < n; ++j) {
        dst[dst_index++] = static_cast<int8_t>(std::round(src[src_index++] * scale));
      }
      memset(&dst[dst_index], 0, pad_n - n);
    } else {  // i >= m; memset;
      for (j = 0; j < shuffle_cols_num; j += shuffle_cols) {
        memset(&dst[dst_index], 0, shuffle_cols);
        dst_index += patch_size;
      }
      memset(&dst[dst_index], 0, pad_n - shuffle_cols_num);
    }
  }
}
}
#endif
