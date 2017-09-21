#ifndef NN_CONVOLUTION_OP_H
#define NN_CONVOLUTION_OP_H

#include "base_convolution.h"
#include "shuffle_convolution.h"

#ifdef TIME_PROFILE
#include <chrono>
#endif

// typedef enum CONV_ALGORITHM {SHULLFE_CONV=0} CONV_ALGORITHM;

struct ConvOp {
  ConvOp() : algo_id_(AUTO_SELECT_CONV), algo_(NULL) {
  }

  ~ConvOp() {
    delete algo_;
  }

  void SetupConvolutionParameter(LAYOUT layout, size_t channel_out, size_t channel_in, size_t groups, size_t kernel_h,
                                 size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                 size_t dilation_h, size_t dilation_w, size_t fusion_mask, CONV_ALGORITHM algo) {
    conv_kernel_desc_ = {
        layout,   channel_out, channel_in, groups, channel_out / groups, channel_in / groups, kernel_h,   kernel_w,
        stride_h, stride_w,    pad_h,      pad_w,  dilation_h,           dilation_w,          fusion_mask};
    ChooseAlgo(algo);
  }

  void SetupConvolutionDataParameter(size_t batch_size, size_t channel_in, size_t height_in, size_t width_in) {
    conv_data_desc_ = {batch_size, channel_in, height_in, width_in};
  }

  void ChooseAlgo(CONV_ALGORITHM algo_id) {
    algo_id_ = algo_id;
    switch (algo_id_) {
      case SHUFFLE_CONV: {
        algo_ = new ShuffleConvolutionAlgo(conv_kernel_desc_);
        break;
      }
      default: {
        algo_ = new ShuffleConvolutionAlgo(conv_kernel_desc_);
        break;
      }
    }
  }

  void InitWeight(float *weight) {
    algo_->InitWeight(weight, conv_kernel_desc_);
  }

  void Execute(float *out, float *data, float *bias, size_t batch_size, size_t channel_in, size_t height_in,
               size_t width_in) {
    SetupConvolutionDataParameter(batch_size, channel_in, height_in, width_in);
    algo_->Execute(out, data, bias, conv_data_desc_, conv_kernel_desc_);
  }

  CONV_ALGORITHM algo_id_;
  BaseConvolutionAlgo *algo_;
  ConvolutionKernelDesc conv_kernel_desc_;
  ConvolutionDataDesc conv_data_desc_;
};
#endif
