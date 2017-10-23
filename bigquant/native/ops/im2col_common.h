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

#ifndef OPS_IM2COL_COMMON_H
#define OPS_IM2COL_COMMON_H

template <typename DType, LAYOUT layout>
INLINE_SPECIFIER void INLINE_ATTRIBUTE FindMinMaxAlongChannel(DType *src, size_t groups, DType *min[], DType *max[],
                                                              size_t batch_size, size_t channels_per_group, size_t h_w,
                                                              DType *transposed_data) {
  assert((layout == NCHW) || (layout == NHWC));
  size_t featuremap_per_image = groups * channels_per_group * h_w;
  if (layout == NCHW) {
    size_t featuremap_per_group = channels_per_group * h_w;
#pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t g = 0; g < groups; ++g) {
        for (size_t s = 0; s < h_w; ++s) {
          DType local_min = FLT_MAX;
          DType local_max = -FLT_MAX;
          size_t dst_index = b * h_w + s;
          size_t src_index = b * featuremap_per_image + g * featuremap_per_group + s;
          for (size_t c = 0; c < channels_per_group; ++c) {
            local_max = fmaxf(local_max, src[src_index]);
            local_min = fminf(local_min, src[src_index]);
            src_index = src_index + h_w;
          }
          max[g][dst_index] = local_max;
          min[g][dst_index] = local_min;
        }
      }
    }
  } else {
#pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t s = 0; s < h_w; ++s) {
        for (size_t g = 0; g < groups; ++g) {
          DType local_min = FLT_MAX;
          DType local_max = -FLT_MAX;
          size_t src_index = b * featuremap_per_image + (s * groups + g) * channels_per_group;
          size_t dst_index = b * h_w + s;
          for (size_t c = 0; c < channels_per_group; ++c) {
            local_max = fmaxf(local_max, src[src_index]);
            local_min = fminf(local_min, src[src_index]);
            ++src_index;
          }
          //  FindMinMaxValue<DType>(src + src_index, channels_per_group, local_min, local_max);
          max[g][dst_index] = local_max;
          min[g][dst_index] = local_min;
        }
      }
    }
  }
}

template <typename DType, LAYOUT layout>
INLINE_SPECIFIER void INLINE_ATTRIBUTE FindMinMaxAlongChannelThenTranspose(DType *&src, size_t groups, DType *min[],
                                                                           DType *max[], size_t batch_size,
                                                                           size_t channels_per_group, size_t h_w,
                                                                           DType *transposed_data) {
  size_t featuremap_per_image = groups * channels_per_group * h_w;
  size_t featuremap_per_group = channels_per_group * h_w;
#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t g = 0; g < groups; ++g) {
      for (size_t s = 0; s < h_w; ++s) {
        DType local_min = FLT_MAX;
        DType local_max = -FLT_MAX;
        size_t dst_index = b * h_w + s;
        size_t src_index = b * featuremap_per_image + g * featuremap_per_group + s;
        size_t total_channels = groups * channels_per_group;
        size_t transposed_index = b * featuremap_per_image + s * total_channels + g * channels_per_group;
        for (size_t c = 0; c < channels_per_group; ++c) {
          local_max = fmaxf(local_max, src[src_index]);
          local_min = fminf(local_min, src[src_index]);
          // Transpose silently and Hope that we can hide this transpose cost.
          transposed_data[transposed_index + c] = src[src_index];
          src_index = src_index + h_w;
        }
        max[g][dst_index] = local_max;
        min[g][dst_index] = local_min;
      }
    }
  }
  // assgin workspace to transposed data
  src = transposed_data;
}

#endif
