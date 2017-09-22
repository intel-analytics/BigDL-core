#ifndef OPS_KENREL_COMMON_H
#define OPS_KENREL_COMMON_H
#include "../base.h"

template <typename DType, size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NHWCGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                    size_t valid_n, size_t i_index, size_t j_index,
                                                                    size_t cur_group, size_t channel_per_group,
                                                                    size_t total_channels) {
  for (size_t kx = 0; kx < std::min(valid_m - i_index, kernel_m); ++kx) {
    size_t c = (i_index + kx) + cur_group * channel_per_group;
    for (size_t ky = 0; ky < std::min(valid_n - j_index, kernel_n); ++ky) {
      result[kx * kernel_n + ky] = pc + (j_index + ky) * total_channels + c;
    }
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NHWCGenrateBlockTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                         size_t valid_n, size_t i_index, size_t j_index,
                                                                         size_t cur_group, size_t channel_per_group,
                                                                         size_t total_channels) {
  size_t c = i_index + cur_group * channel_per_group;
  for (size_t ky = 0; ky < std::min(valid_n - j_index, kernel_n); ++ky) {
    result[ky * kernel_m] = pc + (j_index + ky) * total_channels + c;
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NCHWGenrateTargetAddr(DType *result[], DType *pc, size_t valid_m,
                                                                    size_t valid_n, size_t i_index, size_t j_index,
                                                                    size_t cur_group, size_t feature_map_size_per_image,
                                                                    size_t feature_map_size_per_group,
                                                                    size_t feature_map_size_per_channel) {
  for (size_t kx = 0; kx < kernel_m; ++kx) {
    size_t channel_out = i_index + kx;
    for (size_t ky = 0; ky < kernel_n; ++ky) {
      size_t b = (j_index + ky) / feature_map_size_per_channel;
      size_t h_w = (j_index + ky) % feature_map_size_per_channel;
      size_t index = b * feature_map_size_per_image + cur_group * feature_map_size_per_group +
                     channel_out * feature_map_size_per_channel + h_w;
      result[kx * kernel_n + ky] = pc + index;
    }
  }
}

template <typename DType, size_t kernel_m, size_t kernel_n>
static INLINE_SPECIFIER void INLINE_ATTRIBUTE NCHWGenrateBlockTargetAddr(
    DType *result[], DType *pc, size_t valid_m, size_t valid_n, size_t i_index, size_t j_index, size_t cur_group,
    size_t feature_map_size_per_image, size_t feature_map_size_per_group, size_t feature_map_size_per_channel) {
  size_t b0 = j_index / feature_map_size_per_channel;
  for (size_t kx = 0; kx < kernel_m; ++kx) {
    size_t channel_out = i_index + kx;
    size_t h_w = j_index % feature_map_size_per_channel;
    size_t index = b0 * feature_map_size_per_image + cur_group * feature_map_size_per_group +
                   channel_out * feature_map_size_per_channel + h_w;
    result[kx * kernel_n] = pc + index;
  }
}

// The following function should adapt to AVX512, AVX2 and SSE42
static INLINE_SPECIFIER void INLINE_ATTRIBUTE PRELU(SIMDPSTYPE &result, const SIMDPSTYPE &threshold) {
  result = MAX_PS(result, threshold);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE BN(SIMDPSTYPE &result, const SIMDPSTYPE &global_mean,
                                                 const SIMDPSTYPE &mul_variance_coeff, const SIMDPSTYPE &scale,
                                                 const SIMDPSTYPE &shift) {
  // sub global mean
  result = SUB_PS(result, global_mean);
  // normalized to variance
  result = MUL_PS(result, mul_variance_coeff);
  // Scale Layer
  result = FMA_PS(result, scale, shift);
}

static INLINE_SPECIFIER void INLINE_ATTRIBUTE ScalarBN(float &result, const float &global_mean,
                                                       const float &mul_variance_coeff, const float &scale,
                                                       const float &shift) {
  result -= global_mean;
  result *= mul_variance_coeff;
  result *= scale;
  result += shift;
}

#endif
