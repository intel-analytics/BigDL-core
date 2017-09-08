#ifndef OPS_GROUP_H
#define OPS_GROUP_H

#include "../base.h"

template <typename DType, LAYOUT layout>
void UnGroupKernel(DType* dst[], DType *src, size_t group, size_t channel_out, size_t channel_in, size_t hxw) {
  assert((layout == NCHW) || (layout == NHWC));
  size_t channel_out_per_group = channel_out / group;
  size_t channel_in_per_group = channel_in / group;
  if (layout == NCHW) {
    // for NCHW, there's no need to ungroup kernel
    assert(false);
  } else { // NHWC
    #pragma omp parallel for proc_bind(close) collapse(3)
    for (size_t g = 0; g < group; ++g) {
      for (size_t c_out = 0; c_out < channel_out_per_group; ++c_out) {
        for (size_t i = 0; i < hxw; ++i) {
          size_t dst_index = c_out * hxw * channel_in_per_group + i * channel_in_per_group;
          size_t src_index = (g * channel_out_per_group + c_out) * hxw * channel_in_per_group + i * channel_in_per_group;
          std::memcpy(dst[g] + dst_index, src + src_index, sizeof(DType) * channel_in_per_group);
        }
      }
    }
  }
}
#endif
