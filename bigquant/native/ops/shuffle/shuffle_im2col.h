#ifndef OPS_SHUFFLE_SHUFFLE_IM2COL_H
#define OPS_SHUFFLE_SHUFFLE_IM2COL_H
#include "../../base.h"
#include "../ops.h"
#include "../im2col_common.h"

namespace shuffle {

template <typename DType, size_t shuffle_rows, size_t shuffle_cols, LAYOUT layout>
void PadQuantizeShuffleIm2colRef(DType *data, size_t batch_size,
                                 size_t channels,  // Now No Group Support
                                 size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                 size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                 uint8_t *data_col, DType *min, DType *max, DType *ratio, float sw_threshold = 255.0f) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t patch_size = kernel_h * kernel_w * channels;
  size_t srcsize_per_batch = channels * height * width;
  size_t dstsize_per_batch = patch_size * output_h * output_w;
  // Allocate some tmp memory
  uint8_t *tmp;
  aligned_malloc(reinterpret_cast<void **>(&tmp), 64, batch_size * dstsize_per_batch);
  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t src_index = srcsize_per_batch * batch;
    size_t dst_index = dstsize_per_batch * batch;
    QuantizeIm2col<DType, layout>(data + src_index, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
                                  stride_w, dilation_h, dilation_w, tmp + dst_index, min + batch * output_h * output_w,
                                  max + batch * output_h * output_w, ratio + batch * output_h * output_w, sw_threshold);
  }
  shuffle::PadShuffle2D<uint8_t, shuffle_rows, shuffle_cols>(data_col, batch_size * output_h * output_w, patch_size,
                                                             tmp);
  aligned_free(tmp);
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols, size_t kernel_h, size_t kernel_w>
void PadQuantizeShuffleNCHWIm2col(DType *data, size_t batch_size, size_t channels_per_group, size_t groups,
                                  size_t height, size_t width, size_t pad_h, size_t pad_w, size_t stride_h,
                                  size_t stride_w, size_t dilation_h, size_t dilation_w, uint8_t *data_col[],
                                  DType *min[], DType *max[], DType *ratio[], float sw_threshold) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t kernel_size = kernel_h * kernel_w;
  size_t input_size_per_channel = height * width;
  size_t patch_size = channels_per_group * kernel_size;
  size_t pad_patch_size = GetAlignmentLength(patch_size, shuffle_cols);  // Get Pad Size
  size_t pad_output_spatial_size = GetAlignmentLength(batch_size * output_h * output_w, shuffle_rows);
  std::vector<DType *> min_per_channel(groups);
  std::vector<DType *> max_per_channel(groups);
  for (size_t g = 0; g < groups; ++g) {
    // fix me, hard code is too bad
    aligned_malloc(reinterpret_cast<void **>(&min_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
    aligned_malloc(reinterpret_cast<void **>(&max_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
  }
  FindMinMaxAlongChannel<DType, NCHW>(data, groups, min_per_channel.data(), max_per_channel.data(), batch_size,
                                      channels_per_group, height * width, NULL);
#pragma omp parallel for collapse(2)
  for (size_t batch = 0; batch < batch_size; ++batch) {                   // total batch size
    for (size_t o_y = 0; o_y < output_h; ++o_y) {    // total output rows
      for (size_t o_x = 0; o_x < output_w; ++o_x) {  // total output cols
        // index of output cols
        size_t out_spatial_id = batch * output_h * output_w + o_y * output_w + o_x;
        // name is weird but go on
        size_t col_block = out_spatial_id / shuffle_rows;
        size_t offset_in_block = (out_spatial_id % shuffle_rows) * shuffle_cols;
        int conv_window_y = -pad_h + o_y * stride_h;  // startline of input rows
        int conv_window_x = -pad_w + o_x * stride_w;  // startline of input cols
        for (size_t g = 0; g < groups; ++g) {            // IT Mat Hurt Performance
          uint8_t *addr =
              data_col[g] + col_block * pad_patch_size * shuffle_rows + offset_in_block;  // Get Destination Address
          DType local_min = FLT_MAX;
          DType local_max = -FLT_MAX;
          for (size_t y = 0; y < kernel_h; ++y) {
            int in_y = conv_window_y + y * dilation_h;
            for (size_t x = 0; x < kernel_w; ++x) {
              int in_x = conv_window_x + x * dilation_w;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                local_max =
                    fmaxf(max_per_channel[g][batch * height * width + in_y * width + in_x], local_max);
                local_min =
                    fminf(min_per_channel[g][batch * height * width + in_y * width + in_x], local_min);
              } else {
                DType value = 0;
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              }
            }
          }
          DType scale = sw_threshold / (local_max - local_min);
          min[g][out_spatial_id] = local_min;
          max[g][out_spatial_id] = local_max;
          ratio[g][out_spatial_id] = 1.0f / scale;
          DType shift = -local_min * scale;
          uint8_t zerofill = static_cast<uint8_t>(std::round((shift)));
          // The following code is for NCHW
          int src_base_index =
              batch * channels_per_group * groups * height * width + g * channels_per_group * input_size_per_channel;
          for (size_t c = 0; c < channels_per_group; ++c) {  // total channel && real start of one patch
            size_t channel_offset = src_base_index + c * input_size_per_channel;
            size_t offset = c * kernel_size / shuffle_cols * (shuffle_rows * shuffle_cols);
            offset += (c * kernel_size) % shuffle_cols;
            for (size_t h = 0; h < kernel_h; ++h) {  // total kernel height
              int in_y = conv_window_y + h * dilation_h;
              size_t y_offset = channel_offset + in_y * width;
              for (size_t w = 0; w < kernel_w; ++w) {  // total kernel width
                int in_x = conv_window_x + w * dilation_w;
                if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                  *(addr + offset++) =
                      static_cast<uint8_t>(std::round((data[y_offset + in_x] - local_min) * scale));
                } else {
                  *(addr + offset++) = zerofill;
                }
                if ((offset % shuffle_cols) == 0) {
                  offset += (shuffle_rows - 1) * shuffle_cols;
                }
              }
            }
          }
          // the above code is for NCHW only
          size_t offset = pad_patch_size * shuffle_rows - (shuffle_cols * shuffle_rows) + patch_size % shuffle_cols;
          memset(addr + offset, 0, pad_patch_size - patch_size);
        }
      }
    }
  }
#pragma omp parallel for
  for (size_t i = batch_size * output_h * output_w; i < pad_output_spatial_size; ++i) {
    for (size_t g = 0; g < groups; ++g) {
      size_t col_block = i / shuffle_rows;
      size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
      uint8_t *addr =
          data_col[g] + col_block * shuffle_rows * pad_patch_size + offset_in_block;  // Get Destination Address
      size_t offset = 0;
      for (size_t j = 0; j < pad_patch_size; j += shuffle_cols) {
        memset(addr + offset, 0, shuffle_cols);
        offset += shuffle_cols * shuffle_rows;
      }
    }
  }
  for (size_t g = 0; g < groups; ++g) {
    aligned_free(min_per_channel[g]);
    aligned_free(max_per_channel[g]);
  }
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols>
void PadQuantizeShuffleNCHWIm2col(DType *data, size_t batch_size, size_t channels_per_group, size_t groups,
                                  size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                  size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                  uint8_t *data_col[], DType *min[], DType *max[], DType *ratio[], float sw_threshold) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t kernel_size = kernel_h * kernel_w;
  size_t input_size_per_channel = height * width;
  size_t patch_size = channels_per_group * kernel_size;
  size_t pad_patch_size = GetAlignmentLength(patch_size, shuffle_cols);  // Get Pad Size
  size_t pad_output_spatial_size = GetAlignmentLength(batch_size * output_h * output_w, shuffle_rows);
  std::vector<DType *> min_per_channel(groups);
  std::vector<DType *> max_per_channel(groups);
  for (size_t g = 0; g < groups; ++g) {
    // fix me, hard code is too bad
    aligned_malloc(reinterpret_cast<void **>(&min_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
    aligned_malloc(reinterpret_cast<void **>(&max_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
  }
  FindMinMaxAlongChannel<DType, NCHW>(data, groups, min_per_channel.data(), max_per_channel.data(), batch_size,
                                      channels_per_group, height * width, NULL);
#pragma omp parallel for collapse(3)
  for (size_t batch = 0; batch < batch_size; ++batch) {                   // total batch size
    for (size_t o_y = 0; o_y < output_h; ++o_y) {    // total output rows
      for (size_t o_x = 0; o_x < output_w; ++o_x) {  // total output cols
        // index of output cols
        size_t out_spatial_id = batch * output_h * output_w + o_y * output_w + o_x;
        // name is weird but go on
        size_t col_block = out_spatial_id / shuffle_rows;
        size_t offset_in_block = (out_spatial_id % shuffle_rows) * shuffle_cols;
        int conv_window_y = -pad_h + o_y * stride_h;  // startline of input rows
        int conv_window_x = -pad_w + o_x * stride_w;  // startline of input cols
        for (size_t g = 0; g < groups; ++g) {            // IT Mat Hurt Performance
          uint8_t *addr =
              data_col[g] + col_block * pad_patch_size * shuffle_rows + offset_in_block;  // Get Destination Address
          DType local_min = FLT_MAX;
          DType local_max = -FLT_MAX;
          for (size_t y = 0; y < kernel_h; ++y) {
            int in_y = conv_window_y + y * dilation_h;
            for (size_t x = 0; x < kernel_w; ++x) {
              int in_x = conv_window_x + x * dilation_w;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                local_max =
                    fmaxf(max_per_channel[g][batch * height * width + in_y * width + in_x], local_max);
                local_min =
                    fminf(min_per_channel[g][batch * height * width + in_y * width + in_x], local_min);
              } else {
                DType value = 0;
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              }
            }
          }
          DType scale = sw_threshold / (local_max - local_min);
          min[g][out_spatial_id] = local_min;
          max[g][out_spatial_id] = local_max;
          ratio[g][out_spatial_id] = 1.0f / scale;
          DType shift = -local_min * scale;
          uint8_t zerofill = static_cast<uint8_t>(std::round(shift));
          // The following code is for NCHW
          int src_base_index =
              batch * channels_per_group * groups * height * width + g * channels_per_group * input_size_per_channel;
          for (size_t c = 0; c < channels_per_group; ++c) {  // total channel && real start of one patch
            size_t channel_offset = src_base_index + c * input_size_per_channel;
            size_t offset = c * kernel_size / shuffle_cols * (shuffle_rows * shuffle_cols);
            offset += (c * kernel_size) % shuffle_cols;
            for (size_t h = 0; h < kernel_h; ++h) {  // total kernel height
              int in_y = conv_window_y + h * dilation_h;
              size_t y_offset = channel_offset + in_y * width;
              for (size_t w = 0; w < kernel_w; ++w) {  // total kernel width
                int in_x = conv_window_x + w * dilation_w;
                if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                  *(addr + offset++) =
                      static_cast<uint8_t>(std::round((data[y_offset + in_x] - local_min) * scale));
                } else {
                  *(addr + offset++) = zerofill;
                }
                if ((offset % shuffle_cols) == 0) {
                  offset += (shuffle_rows - 1) * shuffle_cols;
                }
              }
            }
          }
          // the above code is for NCHW only
          size_t offset = pad_patch_size * shuffle_rows - (shuffle_cols * shuffle_rows) + patch_size % shuffle_cols;
          memset(addr + offset, 0, pad_patch_size - patch_size);
        }
      }
    }
  }
#pragma omp parallel for
  for (size_t i = batch_size * output_h * output_w; i < pad_output_spatial_size; ++i) {
    for (size_t g = 0; g < groups; ++g) {
      size_t col_block = i / shuffle_rows;
      size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
      uint8_t *addr =
          data_col[g] + col_block * shuffle_rows * pad_patch_size + offset_in_block;  // Get Destination Address
      size_t offset = 0;
      for (size_t j = 0; j < pad_patch_size; j += shuffle_cols) {
        memset(addr + offset, 0, shuffle_cols);
        offset += shuffle_cols * shuffle_rows;
      }
    }
  }
  for (size_t g = 0; g < groups; ++g) {
    aligned_free(min_per_channel[g]);
    aligned_free(max_per_channel[g]);
  }
}

template <typename DType, size_t shuffle_rows, size_t shuffle_cols, typename findextreme_function,
          typename quantizekernel_function>
void PadQuantizeShuffleNHWCIm2col(DType *data, size_t batch_size, size_t channels_per_group, size_t groups,
                                  size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                  size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w,
                                  uint8_t *data_col[], DType *min[], DType *max[], DType *ratio[], DType *workspace,
                                  float sw_threshold, findextreme_function findextreme,
                                  quantizekernel_function quantizekernel) {
  size_t output_h = GetConvOutSize(height, kernel_h, stride_h, pad_h, dilation_h);
  size_t output_w = GetConvOutSize(width, kernel_w, stride_w, pad_w, dilation_w);
  size_t kernel_size = kernel_h * kernel_w;
  size_t total_channels = groups * channels_per_group;
  size_t input_feature_size_per_batch = height * width * total_channels;
  size_t input_feature_size_per_height = width * total_channels;
  size_t input_feature_size_per_width = total_channels;

  size_t patch_size = channels_per_group * kernel_size;
  size_t pad_patch_size = GetAlignmentLength(patch_size, shuffle_cols);  // Get Pad Size
  size_t pad_output_spatial_size = GetAlignmentLength(batch_size * output_h * output_w, shuffle_rows);
  std::vector<DType *> min_per_channel(groups);
  std::vector<DType *> max_per_channel(groups);
  for (size_t g = 0; g < groups; ++g) {
    aligned_malloc(reinterpret_cast<void **>(&min_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
    aligned_malloc(reinterpret_cast<void **>(&max_per_channel[g]), 64, sizeof(DType) * batch_size * height * width);
  }
#ifdef TIME_PROFILE
  auto start = std::chrono::system_clock::now();
#endif
  findextreme(data, groups, min_per_channel.data(), max_per_channel.data(), batch_size, channels_per_group,
              height * width, workspace);
#ifdef TIME_PROFILE
  auto end = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cerr << "im2col findextreme per channel " << diff.count() << "us" << std::endl;
#endif
#pragma omp parallel for collapse(3)
  for (size_t batch = 0; batch < batch_size; ++batch) {                   // total batch size
    for (size_t o_y = 0; o_y < output_h; ++o_y) {    // total output rows
      for (size_t o_x = 0; o_x < output_w; ++o_x) {  // total output cols
        // index of output cols
        size_t out_spatial_id = batch * output_h * output_w + o_y * output_w + o_x;
        // name is weird but go on
        size_t col_block = out_spatial_id / shuffle_rows;
        size_t offset_in_block = (out_spatial_id % shuffle_rows) * shuffle_cols;
        size_t base_offset = col_block * pad_patch_size * shuffle_rows + offset_in_block;
        int conv_window_y = -pad_h + o_y * stride_h;  // startline of input rows
        int conv_window_x = -pad_w + o_x * stride_w;  // startline of input cols
        size_t batch_offset = batch * height * width;
        for (size_t g = 0; g < groups; ++g) {  // Get min && max && ratio
          DType local_min = FLT_MAX;
          DType local_max = -FLT_MAX;
          for (size_t y = 0; y < kernel_h; ++y) {
            int in_y = conv_window_y + y * dilation_h;
            for (size_t x = 0; x < kernel_w; ++x) {
              int in_x = conv_window_x + x * dilation_w;
              if (x_ge_0_and_x_lt_bound(in_y, height) && x_ge_0_and_x_lt_bound(in_x, width)) {
                local_max = fmaxf(max_per_channel[g][batch_offset + in_y * width + in_x], local_max);
                local_min = fminf(min_per_channel[g][batch_offset + in_y * width + in_x], local_min);
              } else {
                DType value = 0;
                local_max = fmaxf(value, local_max);
                local_min = fminf(value, local_min);
              }
            }
          }
          DType scale = sw_threshold / (local_max - local_min);
          min[g][out_spatial_id] = local_min;
          max[g][out_spatial_id] = local_max;
          ratio[g][out_spatial_id] = 1.0f / scale;
          /* why here shift doesn't plusto 0.5
          * It seems that when converting sse/simd FP32 to Int32, the default mode is round to nearest. So there's no
          * need to plus 0.5 here
          */
          DType shift = -local_min * scale;
          uint8_t zerofill = static_cast<uint8_t>(shift);
          uint8_t *addr = data_col[g] + base_offset;
          size_t src_base_index = batch * input_feature_size_per_batch;
          SIMDPSTYPE simdscale = SET1_PS(scale);
          SIMDPSTYPE simdshift = SET1_PS(shift);
          for (size_t h = 0; h < kernel_h; ++h) {
            int in_y = conv_window_y + h * dilation_h;
            size_t y_offset = src_base_index + in_y * input_feature_size_per_height;
            bool valid_row = x_ge_0_and_x_lt_bound(in_y, height);
            if ((dilation_w == 1) && valid_row && (groups == 1) && x_ge_0_and_x_lt_bound(conv_window_x, width) &&
                x_ge_0_and_x_lt_bound(conv_window_x + kernel_w, width)) {
              const size_t offset_in_row = h * kernel_w * channels_per_group;
              const size_t shuffle_col_id = offset_in_row / shuffle_cols;
              const size_t shuffle_col_remain_index = offset_in_row % shuffle_cols;
              size_t shuffle_offset_in_row =
                  shuffle_col_id * (shuffle_rows * shuffle_cols) + shuffle_col_remain_index;
              size_t src_index = y_offset + conv_window_x * input_feature_size_per_width;
              size_t length = kernel_w * channels_per_group;
              size_t z = 0;
              size_t remain =
                  (shuffle_col_remain_index == 0) ? 0 : std::min(shuffle_cols - shuffle_col_remain_index, length);
              for (; z < remain; ++z) {
                *(addr + shuffle_offset_in_row++) = static_cast<uint8_t>(data[src_index + z] * scale + shift);
                if ((shuffle_offset_in_row % shuffle_cols) == 0) {
                  shuffle_offset_in_row += (shuffle_rows - 1) * shuffle_cols;
                }
              }
              size_t total_kernel = (length - remain) / shuffle_cols;
              DType *src_base = data + src_index + z;
              uint8_t *dst_base = addr + shuffle_offset_in_row;
              for (size_t k = 0; k < total_kernel; ++k) {
                quantizekernel(dst_base + k * shuffle_rows * shuffle_cols, src_base + k * shuffle_cols, simdscale,
                               simdshift);
              }
              shuffle_offset_in_row += total_kernel * shuffle_rows * shuffle_cols;
              z += total_kernel * shuffle_cols;
              for (z = remain + (length - remain) / shuffle_cols * shuffle_cols; z < length; ++z) {
                *(addr + shuffle_offset_in_row++) = static_cast<uint8_t>(data[src_index + z] * scale + shift);
              }
            } else {
              for (size_t w = 0; w < kernel_w; ++w) {
                int in_x = conv_window_x + w * dilation_w;
                size_t x_offset = y_offset + in_x * input_feature_size_per_width;
                bool valid_col = x_ge_0_and_x_lt_bound(in_x, width);
                const size_t offset_in_row = (h * kernel_w + w) * channels_per_group;
                const size_t shuffle_col_id = offset_in_row / shuffle_cols;
                const size_t shuffle_col_remain_index = offset_in_row % shuffle_cols;
                size_t shuffle_offset_in_row =
                    shuffle_col_id * (shuffle_rows * shuffle_cols) + shuffle_col_remain_index;
                if (valid_row && valid_col) {
                  size_t src_index = x_offset + g * channels_per_group;
                  if (channels_per_group < shuffle_cols) {
                    if ((shuffle_col_remain_index + channels_per_group) < shuffle_cols) {
                      for (size_t c = 0; c < channels_per_group; ++c) {
                        *(addr + shuffle_offset_in_row + c) = static_cast<uint8_t>(data[src_index + c] * scale + shift);
                      }
                      shuffle_offset_in_row += channels_per_group;
                    } else {
                      for (size_t c = 0; c < channels_per_group; ++c) {
                        *(addr + shuffle_offset_in_row++) = static_cast<uint8_t>(data[src_index + c] * scale + shift);
                        if ((shuffle_offset_in_row % shuffle_cols) == 0) {
                          shuffle_offset_in_row += (shuffle_rows - 1) * shuffle_cols;
                        }
                      }
                    }
                  } else {
                    size_t c = 0;
                    size_t remain = (shuffle_col_remain_index == 0)
                                        ? 0
                                        : std::min(shuffle_cols - shuffle_col_remain_index, channels_per_group);
                    for (; c < remain; ++c) {
                      *(addr + shuffle_offset_in_row++) = static_cast<uint8_t>(data[src_index + c] * scale + shift);
                      if ((shuffle_offset_in_row % shuffle_cols) == 0) {
                        shuffle_offset_in_row += (shuffle_rows - 1) * shuffle_cols;
                      }
                    }
                    size_t total_kernel = (channels_per_group - remain) / shuffle_cols;
                    DType *src_base = data + src_index + c;
                    uint8_t *dst_base = addr + shuffle_offset_in_row;
                    for (size_t k = 0; k < total_kernel; ++k) {
                      quantizekernel(dst_base + k * shuffle_rows * shuffle_cols, src_base + k * shuffle_cols, simdscale,
                                     simdshift);
                    }
                    shuffle_offset_in_row += total_kernel * shuffle_rows * shuffle_cols;
                    c += total_kernel * shuffle_cols;
                    for (c = remain + (channels_per_group - remain) / shuffle_cols * shuffle_cols;
                         c < channels_per_group; ++c) {
                      *(addr + shuffle_offset_in_row++) = static_cast<uint8_t>(data[src_index + c] * scale + shift);
                    }
                  }
                } else {
                  size_t c = 0;
                  size_t remain = (shuffle_col_remain_index == 0)
                                      ? 0
                                      : std::min(shuffle_cols - shuffle_col_remain_index, channels_per_group);
                  for (; c < remain; ++c) {
                    *(addr + shuffle_offset_in_row++) = zerofill;
                    if ((shuffle_offset_in_row % shuffle_cols) == 0) {
                      shuffle_offset_in_row += (shuffle_rows - 1) * shuffle_cols;
                    }
                  }
                  for (; c < (channels_per_group - remain) / shuffle_cols * shuffle_cols; c += shuffle_cols) {
                    memset(addr + shuffle_offset_in_row, zerofill, shuffle_cols);
                    shuffle_offset_in_row += shuffle_rows * shuffle_cols;
                  }
                  for (c = remain + (channels_per_group - remain) / shuffle_cols * shuffle_cols; c < channels_per_group;
                       ++c) {
                    *(addr + shuffle_offset_in_row++) = zerofill;
                  }
                }
              }
            }
          }
          size_t shuffle_offset_in_row =
              pad_patch_size * shuffle_rows - (shuffle_cols * shuffle_rows) + patch_size % shuffle_cols;
          memset(data_col[g] + base_offset + shuffle_offset_in_row, 0, pad_patch_size - patch_size);
        }
      }
    }
  }

#pragma omp parallel for
  for (size_t i = batch_size * output_h * output_w; i < pad_output_spatial_size; ++i) {
    for (size_t g = 0; g < groups; ++g) {
      size_t col_block = i / shuffle_rows;
      size_t offset_in_block = (i % shuffle_rows) * shuffle_cols;
      uint8_t *addr =
          data_col[g] + col_block * shuffle_rows * pad_patch_size + offset_in_block;  // Get Destination Address
      size_t offset = 0;
      for (size_t j = 0; j < pad_patch_size; j += shuffle_cols) {
        memset(addr + offset, 0, shuffle_cols);
        offset += shuffle_cols * shuffle_rows;
      }
    }
  }
  for (size_t g = 0; g < groups; ++g) {
    aligned_free(min_per_channel[g]);
    aligned_free(max_per_channel[g]);
  }
}

template <typename DType, LAYOUT layout>
void PadQuantizeShuffleIm2colWrapper(DType *data, size_t batch_size, size_t channels_per_group, size_t groups,
                                     size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t pad_h,
                                     size_t pad_w, size_t stride_h, size_t stride_w, size_t dilation_h,
                                     size_t dilation_w, uint8_t *data_col[], DType *min[], DType *max[], DType *ratio[],
                                     DType *workspace, float sw_threshold, bool transpose) {
#if defined(AVX512)
#define QUANTIZE_KERNEL_FUNC AVX512Kernel8Quantize
#elif defined(__AVX2__)
#define QUANTIZE_KERNEL_FUNC AVX2Kernel8Quantize
#else
#define QUANTIZE_KERNEL_FUNC SSE42Kernel16Quantize
#endif

  if (layout == NCHW) {
    if ((kernel_h == 1) && (kernel_w == 1)) {
      PadQuantizeShuffleNCHWIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, 1, 1>(
          data, batch_size, channels_per_group, groups, height, width, pad_h, pad_w, stride_h, stride_w, dilation_h,
          dilation_w, data_col, min, max, ratio, sw_threshold);
    } else if ((kernel_h == 3) && (kernel_w == 3)) {
      PadQuantizeShuffleNCHWIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, 3, 3>(
          data, batch_size, channels_per_group, groups, height, width, pad_h, pad_w, stride_h, stride_w, dilation_h,
          dilation_w, data_col, min, max, ratio, sw_threshold);
    } else if ((kernel_h == 5) && (kernel_w == 5)) {
      PadQuantizeShuffleNCHWIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, 5, 5>(
          data, batch_size, channels_per_group, groups, height, width, pad_h, pad_w, stride_h, stride_w, dilation_h,
          dilation_w, data_col, min, max, ratio, sw_threshold);
    } else {
      PadQuantizeShuffleNCHWIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K>(
          data, batch_size, channels_per_group, groups, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
          stride_w, dilation_h, dilation_w, data_col, min, max, ratio, sw_threshold);
    }
  } else {
    if (transpose == false) {
      PadQuantizeShuffleNHWCIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K>(
          data, batch_size, channels_per_group, groups, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
          stride_w, dilation_h, dilation_w, data_col, min, max, ratio, NULL, sw_threshold,
          FindMinMaxAlongChannel<DType, NHWC>, QUANTIZE_KERNEL_FUNC);
    } else {
      DType *tmp;
      if (workspace == NULL) {
        aligned_malloc(reinterpret_cast<void **>(&tmp), 64,
                       batch_size * groups * channels_per_group * height * width * sizeof(DType));
      } else {
        tmp = workspace;
      }
      PadQuantizeShuffleNHWCIm2col<DType, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K>(
          data, batch_size, channels_per_group, groups, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
          stride_w, dilation_h, dilation_w, data_col, min, max, ratio, tmp, sw_threshold,
          FindMinMaxAlongChannelThenTranspose<DType, NHWC>, QUANTIZE_KERNEL_FUNC);
      if (workspace == NULL) {
        aligned_free(tmp);
      }
    }
  }
}
}
#endif
