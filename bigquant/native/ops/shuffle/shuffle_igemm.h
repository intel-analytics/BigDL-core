#ifndef IGEMM_X64
#define IGEMM_X64

#include "../../base.h"
#include "../../common.h"
#include "../kernel-common.h"
#define UNROLL_NUM 4

#if defined(AVX512)
#include "../kernel/shuffle_avx512_igemm_8x8x8.h"
#elif defined(__AVX2__)
#include "../kernel/shuffle_avx2_sse42_igemm_4xnx8-x64.h"
#include "../kernel/shuffle_avx2_igemm4x1x32.h"
#elif !defined(__AVX2__) && defined(__SSE4_2__)
#include "../kernel/shuffle_sse42_igemm2x2x16.h"
#endif

namespace shuffle {

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, typename GEMM_KERNEL>
void ShuffleGEMM(int8_t *pa, uint8_t *pb, int *pc, size_t m, size_t n, size_t k, float fault_tolerance, size_t pad_m,
                 size_t pad_n, GEMM_KERNEL kernel) {
  assert((fault_tolerance <= 1.0f) && (fault_tolerance >= 0.0f));
  size_t m_in_l1, m_in_l2, m_in_l3, n_in_l1, n_in_l2, n_in_l3;
  GetBlocksInfo<kernel_m>(m, k, m_in_l1, m_in_l2, m_in_l3);
  GetBlocksInfo<kernel_n>(n, k, n_in_l1, n_in_l2, n_in_l3);
  size_t valid_m = m - pad_m;
  size_t valid_n = n - pad_n;
  bool mltn = m < n;
  std::array<size_t, 10> blocks1 = {n, m, n_in_l3, m_in_l3, n_in_l2, m_in_l2, n_in_l1, m_in_l1, kernel_n, kernel_m};
  std::array<size_t, 10> blocks2 = {m, n, m_in_l3, n_in_l3, m_in_l2, n_in_l2, m_in_l1, n_in_l1, kernel_m, kernel_n};
  std::array<size_t, 10> &blocks = (mltn) ? blocks1 : blocks2;
#pragma omp parallel proc_bind(close)
  {
    for (size_t y3 = 0; y3 < blocks[0]; y3 += blocks[2]) {
      for (size_t x3 = 0; x3 < blocks[1]; x3 += blocks[3]) {
#pragma omp for collapse(2) schedule(dynamic) nowait
        for (size_t y2 = 0; y2 < blocks[2]; y2 += blocks[4]) {
          for (size_t x2 = 0; x2 < blocks[3]; x2 += blocks[5]) {
            for (size_t y1 = 0; y1 < blocks[4]; y1 += blocks[6]) {
              for (size_t x1 = 0; x1 < blocks[5]; x1 += blocks[7]) {
                for (size_t y0 = 0; y0 < blocks[6]; y0 += blocks[8]) {
                  for (size_t x0 = 0; x0 < blocks[7]; x0 += blocks[9]) {
                    auto y_sum = y3 + y2 + y1 + y0;
                    auto x_sum = x3 + x2 + x1 + x0;
                    auto j_index = mltn ? y_sum : x_sum;
                    auto i_index = mltn ? x_sum : y_sum;
                    if ((j_index < n) && (i_index < m)) {
                      int8_t *local_pa = pa + i_index * k;
                      uint8_t *local_pb = pb + j_index * k;
                      void *result[kernel_m];
                      for (size_t kx = 0; kx < kernel_m; ++kx) {
                        size_t dst_addr = (i_index + kx) * valid_n + j_index;
                        result[kx] = reinterpret_cast<void *>(pc + dst_addr);
                      }
                      kernel(local_pa, local_pb, k, fault_tolerance, result, std::min(valid_m - i_index, kernel_m),
                             std::min(valid_n - j_index, kernel_n));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Common Convolution. It's one purely gemm which can be used in wider application.
template <size_t kernel_m, size_t kernel_n, size_t kernel_k, typename GEMM_KERNEL>
void InternalMixPrecisionGemm(ORDER order, enum TRANSPOSE transA, enum TRANSPOSE transB, int m, int n, int k, int8_t *a,
                              int lda, uint8_t *b, int ldb, int *c, int ldc, float fault_tolerance,
                              GEMM_KERNEL kernel) {
  assert(order == 101);   // We use RowMajor and only support RowMajor
  assert(transA == 111);  // A is not transposed.
  assert(transB == 112);  // B is transposed.
  size_t m_out = GetAlignmentLength(m, kernel_m);
  size_t k_out = GetAlignmentLength(k, kernel_k);
  size_t n_out = GetAlignmentLength(n, kernel_n);
  int8_t *pad_a;
  uint8_t *pad_b;
  aligned_malloc(reinterpret_cast<void **>(&pad_a), 64, sizeof(int8_t) * m_out * k_out);
  aligned_malloc(reinterpret_cast<void **>(&pad_b), 64, sizeof(uint8_t) * n_out * k_out);
  PadShuffle2D<int8_t, kernel_m, kernel_k>(pad_a, m, k, a);
  PadShuffle2D<uint8_t, kernel_n, kernel_k>(pad_b, n, k, b);
#ifdef TIME_PROFILE
  auto start = std::chrono::system_clock::now();
#endif
  // TODO(yandai) better wrapper
  ShuffleGEMM<kernel_m, kernel_n, kernel_k>(pad_a, pad_b, c, m_out, n_out, k_out, fault_tolerance, m_out - m, n_out - n,
                                            kernel);
#ifdef TIME_PROFILE
  auto end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cerr << std::endl << "time = " << diff.count() << "us" << std::endl;
  std::cerr << "flops = " << (2.0 * m * n * k / (diff.count() / 1.0e6)) / 1.0e9 << std::endl;
#endif
  aligned_free(pad_a);
  aligned_free(pad_b);
}

template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE QuantizedGemmSelect(
    int8_t *&pa, uint8_t *&pb, size_t k, float fault_tolerance, float *result[], size_t length, size_t valid_lanes,
    size_t i_index, size_t j_index, float *ratio_a, float *ratio_b, float *min_b, float *kernel_sum, float *bias,
    bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean,
    float *mul_variance_coeff, float *scale, float *shift, bool is_block) {
#if defined(AVX512)
  if ((kernel_m == 8) && (kernel_n == 8) && (kernel_k == 8)) {
    kernel::avx512_igemm8x8x8::ApplyKernelWrapper<kernel_m, kernel_n, kernel_k, layout>(
        pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
        bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
        mul_variance_coeff, scale, shift, is_block);
  }
#elif defined(__AVX2__)
  if ((kernel_m == 4) && (kernel_n == 8) && (kernel_k == 8)) {
    kernel::igemm4xn::ApplyKernelWrapper<kernel_m, kernel_n, kernel_k, layout>(
        pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
        bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
        mul_variance_coeff, scale, shift, is_block);
  }
  if ((kernel_m == 4) && (kernel_n == 1) && (kernel_k == 32)) {
    kernel::igemm4x1::ApplyKernelWrapper<kernel_m, kernel_n, kernel_k, layout>(
        pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
        bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
        mul_variance_coeff, scale, shift, is_block);
  }
#else
  /*
  if ((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 8)) {
    kernel::igemm4xn::ApplyKernelWrapper<kernel_m, kernel_n, kernel_k, layout>(pa, pb, k, fault_tolerance, result,
  length, valid_lanes, i_index, j_index,
                                                                        ratio_a, ratio_b, min_b, kernel_sum, bias,
                                                                        conv_relu_fusion, conv_bn_fusion,
  conv_bn_relu_fusion, conv_relu_bn_fusion,
                                                                        global_mean, mul_variance_coeff, scale, shift,
                                                                        is_block);
  }
  */
  if ((kernel_m == 2) && (kernel_n == 2) && (kernel_k == 16)) {
    kernel::sse42_igemm2x2x16::ApplyKernelWrapper<kernel_m, kernel_n, kernel_k, layout>(
        pa, pb, k, fault_tolerance, result, length, valid_lanes, i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
        bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
        mul_variance_coeff, scale, shift, is_block);
  }
#endif
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER bool INLINE_ATTRIBUTE NHWCRTGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                      size_t valid_n, size_t i_index, size_t j_index,
                                                                      size_t cur_group, size_t channel_per_group,
                                                                      size_t total_channels) {
#if defined(AVX512)
  if ((kernel_m == 8) && (kernel_n == 8) && (kernel_k == 8)) {
    return kernel::avx512_igemm8x8x8::NHWCRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  }
#elif defined(__AVX2__)
  if ((kernel_m == 4) && (kernel_n == 8) && (kernel_k == 8)) {
    return kernel::igemm4xn::NHWCRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  }
  if ((kernel_m == 4) && (kernel_n == 1) && (kernel_k == 32)) {
    // return kernel::igemm4x1::NHWCRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(result, pc, valid_m,
    // valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  }
#else
  /*
  if ((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 8)) {
    return kernel::igemm4xn::NHWCRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(result, pc, valid_m, valid_n,
  i_index, j_index, cur_group, channel_per_group, total_channels);
  }
  */
  if ((kernel_m == 2) && (kernel_n == 2) && (kernel_k == 16)) {
    return kernel::sse42_igemm2x2x16::NHWCRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group, total_channels);
  }
#endif
}

template <typename DType, size_t kernel_m, size_t kernel_n, size_t kernel_k>
static INLINE_SPECIFIER int INLINE_ATTRIBUTE NCHWRTGenrateTargetAddr(
    DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, size_t cur_group,
    size_t feature_map_size_per_image, size_t feature_map_size_per_group, size_t feature_map_size_per_channel) {
#if defined(AVX512)
  if ((kernel_m == 8) && (kernel_n == 8) && (kernel_k == 8)) {
    return kernel::avx512_igemm8x8x8::NCHWRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
        feature_map_size_per_group, feature_map_size_per_channel);
  }
#elif defined(__AVX2__)
  if ((kernel_m == 4) && (kernel_n == 8) && (kernel_k == 8)) {
    return kernel::igemm4xn::NCHWRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
        feature_map_size_per_group, feature_map_size_per_channel);
  }
  if ((kernel_m == 4) && (kernel_n == 1) && (kernel_k == 32)) {
    return kernel::igemm4x1::NCHWRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
        feature_map_size_per_group, feature_map_size_per_channel);
  }
#else
  /*
  if ((kernel_m == 4) && (kernel_n == 4) && (kernel_k == 8)) {
    return kernel::igemm4xn::NCHWRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(result, pc, valid_m, valid_n,
                                                                                  i_index, j_index, cur_group,
                                                                                  feature_map_size_per_image,
  feature_map_size_per_group,
                                                                                  feature_map_size_per_channel);
  }
  */
  if ((kernel_m == 2) && (kernel_n == 2) && (kernel_k == 16)) {
    return kernel::sse42_igemm2x2x16::NCHWRTGenrateTargetAddr<DType, kernel_m, kernel_n, kernel_k>(
        result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
        feature_map_size_per_group, feature_map_size_per_channel);
  }
#endif
}

#if defined(LLC_SHARED)
template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
void ConvShuffleGEMM(int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k, float *ratio_a, float *ratio_b,
                     float *kernel_sum, float *min_b, float *bias, size_t batch_size, size_t groups,
                     size_t channel_per_group, size_t cur_group, size_t height_out, size_t width_out,
                     float fault_tolerance, size_t pad_m, size_t pad_n, bool conv_relu_fusion, bool conv_bn_fusion,
                     bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean, float *mul_variance_coeff,
                     float *scale, float *shift) {
  assert((fault_tolerance <= 1.0f) && (fault_tolerance >= 0.0f));
  assert((layout == NCHW) || (layout == NHWC));
  size_t feature_map_size_per_channel = height_out * width_out;
  size_t total_channels = channel_per_group * groups;
  size_t feature_map_size_per_image = total_channels * height_out * width_out;
  size_t feature_map_size_per_group = height_out * width_out * channel_per_group;
  size_t m_in_l1, m_in_l2, m_in_l3, n_in_l1, n_in_l2, n_in_l3;
  GetBlocksInfo<kernel_m>(m, k, m_in_l1, m_in_l2, m_in_l3);
  GetBlocksInfo<kernel_n>(n, k, n_in_l1, n_in_l2, n_in_l3);
  size_t valid_m = m - pad_m;
  size_t valid_n = n - pad_n;
  bool mltn = m < n;
  std::array<size_t, 10> blocks1 = {n, m, n_in_l3, m_in_l3, n_in_l2, m_in_l2, n_in_l1, m_in_l1, kernel_n, kernel_m};
  std::array<size_t, 10> blocks2 = {m, n, m_in_l3, n_in_l3, m_in_l2, n_in_l2, m_in_l1, n_in_l1, kernel_m, kernel_n};
  std::array<size_t, 10> &blocks = (mltn) ? blocks1 : blocks2;
#pragma omp parallel proc_bind(close)
  {
    for (size_t y3 = 0; y3 < blocks[0]; y3 += blocks[2]) {
      for (size_t x3 = 0; x3 < blocks[1]; x3 += blocks[3]) {
#pragma omp for collapse(2) schedule(dynamic, 4) nowait
        for (size_t y2 = 0; y2 < blocks[2]; y2 += blocks[4]) {
          for (size_t x2 = 0; x2 < blocks[3]; x2 += blocks[5]) {
            for (size_t y1 = 0; y1 < blocks[4]; y1 += blocks[6]) {
              for (size_t x1 = 0; x1 < blocks[5]; x1 += blocks[7]) {
                for (size_t y0 = 0; y0 < blocks[6]; y0 += blocks[8]) {
                  for (size_t x0 = 0; x0 < blocks[7]; x0 += blocks[9]) {
                    auto y_sum = y3 + y2 + y1 + y0;
                    auto x_sum = x3 + x2 + x1 + x0;
                    auto j_index = mltn ? y_sum : x_sum;
                    auto i_index = mltn ? x_sum : y_sum;
                    if ((j_index < n) && (i_index < m)) {
                      float *result[kernel_m * kernel_n];
                      int8_t *local_pa = pa + i_index * k;
                      uint8_t *local_pb = pb + j_index * k;
                      bool is_block;
                      if (layout == NCHW) {
                        is_block = NCHWRTGenrateTargetAddr<float, kernel_m, kernel_n, kernel_k>(
                            result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
                            feature_map_size_per_group, feature_map_size_per_channel);
                      } else {
                        is_block = NHWCRTGenrateTargetAddr<float, kernel_m, kernel_n, kernel_k>(
                            result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group,
                            total_channels);
                      }
                      QuantizedGemmSelect<kernel_m, kernel_n, kernel_k, layout>(
                          local_pa, local_pb, k, fault_tolerance, result, std::min(valid_m - i_index, kernel_m),
                          std::min(valid_n - j_index, kernel_n), i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
                          bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
                          mul_variance_coeff, scale, shift, is_block);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif
#if defined(LLC_EXCLUSIVE)
template <size_t kernel_m, size_t kernel_n, size_t kernel_k, LAYOUT layout>
void ConvShuffleGEMM(int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k, float *ratio_a, float *ratio_b,
                     float *kernel_sum, float *min_b, float *bias, size_t batch_size, size_t groups,
                     size_t channel_per_group, size_t cur_group, size_t height_out, size_t width_out,
                     float fault_tolerance, size_t pad_m, size_t pad_n, bool conv_relu_fusion, bool conv_bn_fusion,
                     bool conv_bn_relu_fusion, bool conv_relu_bn_fusion, float *global_mean, float *mul_variance_coeff,
                     float *scale, float *shift) {
  assert((fault_tolerance <= 1.0f) && (fault_tolerance >= 0.0f));
  assert((layout == NCHW) || (layout == NHWC));
  size_t feature_map_size_per_channel = height_out * width_out;
  size_t total_channels = channel_per_group * groups;
  size_t feature_map_size_per_image = total_channels * height_out * width_out;
  size_t feature_map_size_per_group = height_out * width_out * channel_per_group;
  size_t m_in_l1, m_in_l2, m_in_l3, n_in_l1, n_in_l2, n_in_l3;
  GetBlocksInfo<kernel_m>(m, k, m_in_l1, m_in_l2, m_in_l3);
  GetBlocksInfo<kernel_n>(n, k, n_in_l1, n_in_l2, n_in_l3);
  size_t valid_m = m - pad_m;
  size_t valid_n = n - pad_n;
  bool mltn = m < n;
  std::array<size_t, 10> blocks1 = {n, m, n_in_l3, m_in_l3, n_in_l2, m_in_l2, n_in_l1, m_in_l1, kernel_n, kernel_m};
  std::array<size_t, 10> blocks2 = {m, n, m_in_l3, n_in_l3, m_in_l2, n_in_l2, m_in_l1, n_in_l1, kernel_m, kernel_n};
  std::array<size_t, 10> &blocks = (mltn) ? blocks1 : blocks2;
  {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t y3 = 0; y3 < blocks[0]; y3 += blocks[2]) {
      for (size_t x3 = 0; x3 < blocks[1]; x3 += blocks[3]) {
        for (size_t y2 = 0; y2 < blocks[2]; y2 += blocks[4]) {
          for (size_t x2 = 0; x2 < blocks[3]; x2 += blocks[5]) {
            for (size_t y1 = 0; y1 < blocks[4]; y1 += blocks[6]) {
              for (size_t x1 = 0; x1 < blocks[5]; x1 += blocks[7]) {
                for (size_t y0 = 0; y0 < blocks[6]; y0 += blocks[8]) {
                  for (size_t x0 = 0; x0 < blocks[7]; x0 += blocks[9]) {
                    auto y_sum = y3 + y2 + y1 + y0;
                    auto x_sum = x3 + x2 + x1 + x0;
                    auto j_index = mltn ? y_sum : x_sum;
                    auto i_index = mltn ? x_sum : y_sum;
                    if ((j_index < n) && (i_index < m)) {
                      float *result[kernel_m * kernel_n];
                      int8_t *local_pa = pa + i_index * k;
                      uint8_t *local_pb = pb + j_index * k;
                      bool is_block;
                      if (layout == NCHW) {
                        is_block = NCHWRTGenrateTargetAddr<float, kernel_m, kernel_n, kernel_k>(
                            result, pc, valid_m, valid_n, i_index, j_index, cur_group, feature_map_size_per_image,
                            feature_map_size_per_group, feature_map_size_per_channel);
                      } else {
                        is_block = NHWCRTGenrateTargetAddr<float, kernel_m, kernel_n, kernel_k>(
                            result, pc, valid_m, valid_n, i_index, j_index, cur_group, channel_per_group,
                            total_channels);
                      }
                      QuantizedGemmSelect<kernel_m, kernel_n, kernel_k, layout>(
                          local_pa, local_pb, k, fault_tolerance, result, std::min(valid_m - i_index, kernel_m),
                          std::min(valid_n - j_index, kernel_n), i_index, j_index, ratio_a, ratio_b, min_b, kernel_sum,
                          bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean,
                          mul_variance_coeff, scale, shift, is_block);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#endif
}
#endif
