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

#ifndef NN_BASE_CONVOLUTION_H
#define NN_BASE_CONVOLUTION_H

#include "../base.h"
#include "../common.h"
#include "../tensor.h"
#include "../ops/ops.h"
#ifdef TIME_PROFILE
#include <chrono>
#endif

struct ConvolutionKernelDesc {
  LAYOUT layout_;

  size_t channel_out_;
  size_t channel_in_;
  size_t group_;
  size_t channel_out_per_group_;
  size_t channel_in_per_group_;
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  size_t dilation_h_;
  size_t dilation_w_;

  size_t fusion_mask_;
};

struct ConvolutionDataDesc {
  size_t batch_size_;
  size_t channel_in_;
  size_t height_in_;
  size_t width_in_;
};

struct BaseConvolutionAlgo {

  BaseConvolutionAlgo() = default;

  BaseConvolutionAlgo(const BaseConvolutionAlgo&) = delete;

  BaseConvolutionAlgo& operator=(const BaseConvolutionAlgo&) = delete;

  virtual ~BaseConvolutionAlgo(){

  };
  virtual void InitWeight(float *weight, ConvolutionKernelDesc &conv_kernel_desc) = 0;
  virtual void Execute(float *out, float *data, float *bias, ConvolutionDataDesc &conv_data_desc,
                       ConvolutionKernelDesc &conv_kernel_desc) = 0;

 protected:
  size_t height_out_;
  size_t width_out_;
};

#endif
