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
