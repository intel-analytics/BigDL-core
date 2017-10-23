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

#ifndef MODEL_H
#define MODEL_H

void DequantizeModel(float *dst, int8_t *src, float *src_min, float *src_max, size_t c_out, size_t c_in,
                     size_t kernel_h, size_t kernel_w) {
  for (size_t c_o = 0; c_o < c_out; ++c_o) {
    size_t meta_index = c_o;
    for (size_t k = 0; k < c_in * kernel_h * kernel_w; ++k) {
      size_t index = c_o * c_in * kernel_h * kernel_w + k;
      dst[index] =
          1.0 * static_cast<float>(src[index]) / 127.0 * fmaxf(fabs(src_max[meta_index]), fabs(src_min[meta_index]));
    }
  }
}
#endif
