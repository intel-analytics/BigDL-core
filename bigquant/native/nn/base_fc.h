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

#ifndef NN_BASE_FC_H
#define NN_BASE_FC_H

#include "../base.h"
#include "../common.h"
#include "../tensor.h"
#include "../ops/ops.h"

struct FCKernelDesc {
  LAYOUT layout_;
  size_t channel_out_;
  size_t channel_in_;
};
struct FCDataDesc {
  size_t batch_size_;
  size_t channel_in_;
};

struct BaseFCAlgo {

  BaseFCAlgo() = default;

  BaseFCAlgo(const BaseFCAlgo&) = delete;

  BaseFCAlgo& operator=(const BaseFCAlgo&) = delete;

  virtual ~BaseFCAlgo() {
  }
  virtual void InitWeight(float *weight, FCKernelDesc &fc_kernel_desc) = 0;
  virtual void Execute(float *out, float *data, float *bias, FCDataDesc &fc_data_desc,
                       FCKernelDesc &fc_kernel_desc) = 0;
};

#endif
