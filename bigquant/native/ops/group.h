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

#ifndef OPS_GROUP_H
#define OPS_GROUP_H

#include "../base.h"

template <typename DType, LAYOUT layout>
void UnGroupKernel(DType* dst[], DType* src, size_t group, size_t channel_out, size_t channel_in, size_t hxw) {
  assert((layout == NCHW) || (layout == NHWC));
  size_t channel_out_per_group = channel_out / group;
  size_t channel_in_per_group = channel_in / group;
  if (layout == NCHW) {
    // for NCHW, there's no need to ungroup kernel
    assert(false);
  } else {  // NHWC
#pragma omp parallel for proc_bind(close) collapse(3)
    for (size_t g = 0; g < group; ++g) {
      for (size_t c_out = 0; c_out < channel_out_per_group; ++c_out) {
        for (size_t i = 0; i < hxw; ++i) {
          size_t dst_index = c_out * hxw * channel_in_per_group + i * channel_in_per_group;
          size_t src_index =
              (g * channel_out_per_group + c_out) * hxw * channel_in_per_group + i * channel_in_per_group;
          std::memcpy(dst[g] + dst_index, src + src_index, sizeof(DType) * channel_in_per_group);
        }
      }
    }
  }
}
#endif
