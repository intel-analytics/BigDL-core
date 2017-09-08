#ifndef OPS_LAYOUT_H
#define OPS_LAYOUT_H

#include "../base.h"

#ifdef TIME_PROFILE
#include <chrono>
#endif

/*
#if defined(MKL_TRANSPOSE)
#include <mkl.h>
template <typename DType>
void TransposeWrapper(const char ordering, const char trans, size_t rows, size_t cols, const DType alpha, DType * AB, size_t lda, size_t ldb) {
  std::cerr << "Unimplemented" << std::endl;
}

template<>
void TransposeWrapper<float>(const char ordering, const char trans, size_t rows, size_t cols, const float alpha, float * AB, size_t lda, size_t ldb) {
  mkl_simatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
}

template<>
void TransposeWrapper<double>(const char ordering, const char trans, size_t rows, size_t cols, const double alpha, double * AB, size_t lda, size_t ldb) {
  mkl_dimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
}

template <typename DType>
void TransformLayoutInPlace(LAYOUT dst_layout, LAYOUT src_layout, DType *src, size_t batch_size, size_t channels, size_t hxw) {
  if ((dst_layout == NHWC) && (src_layout == NCHW)) {
    for (size_t n = 0; n < batch_size; ++n) {
      size_t batch_offset = n * channels * hxw;
      TransposeWrapper<DType>('r', 't', channels, hxw, 1.0f, (src + batch_offset), hxw, channels);
    }
  } else if ((dst_layout == NCHW) && (src_layout == NHWC)) {
    for (size_t n = 0; n < batch_size; ++n) {
      size_t batch_offset = n * channels * hxw;
      TransposeWrapper<DType>('r', 't', hxw, channels, 1.0f, (src + batch_offset), channels, hxw);
    }
  }
}
#endif
*/

template <typename DType>
void Transpose(DType *dst, DType *src, size_t m, size_t n) {
#ifdef TIME_PROFILE
  auto start = std::chrono::system_clock::now();
#endif
  for (size_t y = 0; y < n; ++y) {
    for (size_t x = 0; x < m; ++x) {
      *(dst + y * m + x) = *(src + x * n + y);
    }
  }
#ifdef TIME_PROFILE
  auto end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cerr << "layout transform " << diff.count() << "us" << " " << m << " " << n << std::endl;
#endif
}

template <typename DType>
void TransformLayout(LAYOUT dst_layout, LAYOUT src_layout, DType *dst, DType *src, size_t batch_size, size_t channels, size_t hxw) {
#ifdef TIME_PROFILE
  auto start = std::chrono::system_clock::now();
#endif
  if ((dst_layout == NHWC) && (src_layout == NCHW)) {
    #pragma omp parallel for collapse(2) schedule(static, 1)
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t s = 0; s < hxw; ++s) {
        size_t batch_offset = n * channels * hxw;
        size_t offset = batch_offset + s * channels;
        DType *src_per_pixel = src + batch_offset + s;
        DType *dst_per_pixel = dst + offset;
        for (size_t c = 0; c < channels; ++c) {
          *(dst_per_pixel + c) = *(src_per_pixel + c * hxw);
        }
      }
    }
  } else if ((dst_layout == NCHW) && (src_layout == NHWC)) {
    #pragma omp parallel for collapse(2) schedule(static, 1)
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        size_t batch_offset = n * channels * hxw;
        size_t offset = batch_offset + c * hxw;
        DType *dst_per_channel = dst + offset;
        DType *src_per_channel = src + batch_offset + c;
        for (size_t s = 0; s < hxw; ++s) {
          *(dst_per_channel + s) = *(src_per_channel + s * channels);
        }
      }
    }
  }
#ifdef TIME_PROFILE
  auto end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cerr << "layout transform " << diff.count() << "us" << " " << batch_size << " " << channels << " " << hxw << std::endl;
#endif
}

#endif
