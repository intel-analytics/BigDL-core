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

#ifndef OPS_DOT_H
#define OPS_DOT_H

#include "../base.h"
#include "../common.h"
#include "./kernel/streamdot.h"
#include "./kernel/multistream4x2_dot.h"
namespace dot {

void Dot(int8_t* pa, uint8_t* pb, int& result, size_t length) {
  kernel::dot::ApplyKernel(pa, pb, result, length);
}

void Dot(int8_t* pa, uint8_t* pb, float& result, size_t length, float ratio_a, float a_sum, float ratio_b,
         float min_b) {
  kernel::dot::ApplyKernel(pa, pb, result, length, ratio_a, a_sum, ratio_b, min_b);
}
}

#endif
