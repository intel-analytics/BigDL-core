#ifndef BASECONVOLUTION_H
#define BASECONVOLUTION_H

#include "base.h"
#include "common.h"
#include "tensor.h"
#include "ops/ops.h"
#ifdef TIME_PROFILE
#include <chrono>
#endif


struct BaseConvolutionAlgo {

  BaseConvolutionAlgo(const ConvolutionKernelDesc &conv_kernel_desc)
    : conv_kernel_desc_(conv_kernel_desc) {

  }

  virtual void InitWeight(const float *weight) = 0;
  virtual void Execute(float *out, const float *data, const float *bias, const ConvolutionDataDesc &conv_data_desc) = 0;
  virtual void Free() = 0;


  inline size_t GetConvOutSize(size_t in, size_t kernel, size_t stride, size_t pad, size_t dilation) {
    return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
  }

protected:
  const ConvolutionKernelDesc &conv_kernel_desc_;
  const ConvolutionDataDesc &conv_data_desc_;

  size_t height_out_;
  size_t width_out_;

};
#endif
