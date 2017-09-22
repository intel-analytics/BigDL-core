#ifndef NN_SHUFFLE_FC_H
#define NN_SHUFFLE_FC_H

#include "base_fc.h"

struct ShuffleFCAlgo : public BaseFCAlgo {
  ShuffleFCAlgo() {
    weight_threshold_ = 64.0f;
    data_threshold_ = 127.0f;
  }

  ~ShuffleFCAlgo() {
    if (sum_per_channel_out_) {
      delete sum_per_channel_out_;
      sum_per_channel_out_ = NULL;
    }
    if (quantized_kernel_) {
      delete quantized_kernel_;
      quantized_kernel_ = NULL;
    }
  }

  void InitWeight(float *weight, FCKernelDesc &fc_kernel_desc) {
    fc_m_ = fc_kernel_desc.channel_out_;
    fc_k_ = fc_kernel_desc.channel_in_;
    aligned_fc_m_ = GetAlignmentLength(fc_m_, FC_SHUFFLE_KERNEL_M);
    aligned_fc_k_ = GetAlignmentLength(fc_k_, FC_SHUFFLE_KERNEL_K);

    sum_per_channel_out_ = new Tensor<float>(make_shape(fc_kernel_desc.channel_out_), 64);
    ComputeMatrixSumPerRow<float>(sum_per_channel_out_->data_, weight, fc_kernel_desc.channel_out_,
                                  fc_kernel_desc.channel_in_);
    quantized_kernel_ = new QuantizedTensor<float, int8_t>(make_shape(aligned_fc_m_, aligned_fc_k_), make_shape(fc_m_),
                                                           make_shape(fc_m_, fc_k_), 64);
    shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_K>(
        quantized_kernel_->data_, fc_m_, fc_k_, aligned_fc_m_, aligned_fc_k_, weight, quantized_kernel_->min_.data_,
        quantized_kernel_->max_.data_, quantized_kernel_->ratio_.data_, weight_threshold_);
  }

  void Execute(float *out, float *data, float *bias, FCDataDesc &fc_data_desc, FCKernelDesc &fc_kernel_desc) {
    fc_n_ = fc_data_desc.batch_size_;
    aligned_fc_n_ = GetAlignmentLength(fc_n_, FC_SHUFFLE_KERNEL_N);
    quantized_data_ = new QuantizedTensor<float, uint8_t>(make_shape(aligned_fc_n_, aligned_fc_k_), make_shape(fc_n_),
                                                          make_shape(fc_n_, fc_k_), 64);

    shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K>(
        quantized_data_->data_, fc_n_, fc_k_, aligned_fc_n_, aligned_fc_k_, data, quantized_data_->min_.data_,
        quantized_data_->max_.data_, quantized_data_->ratio_.data_, data_threshold_);
    if (fc_kernel_desc.layout_ == NCHW) {
      shuffle::ConvShuffleGEMM<FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>(
          quantized_kernel_->data_, quantized_data_->data_, out, aligned_fc_m_, aligned_fc_n_, aligned_fc_k_,
          quantized_kernel_->ratio_.data_, quantized_data_->ratio_.data_, sum_per_channel_out_->data_,
          quantized_data_->min_.data_, bias, fc_data_desc.batch_size_, 1, fc_kernel_desc.channel_out_, 0, 1, 1, 0.5,
          aligned_fc_m_ - fc_m_, aligned_fc_n_ - fc_n_, false);
    } else {
      shuffle::ConvShuffleGEMM<FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>(
          quantized_kernel_->data_, quantized_data_->data_, out, aligned_fc_m_, aligned_fc_n_, aligned_fc_k_,
          quantized_kernel_->ratio_.data_, quantized_data_->ratio_.data_, sum_per_channel_out_->data_,
          quantized_data_->min_.data_, bias, fc_data_desc.batch_size_, 1, fc_kernel_desc.channel_out_, 0, 1, 1, 0.5,
          aligned_fc_m_ - fc_m_, aligned_fc_n_ - fc_n_, false);
    }
    delete quantized_data_;
    quantized_data_ = NULL;
  }

 private:
  size_t fc_m_;
  size_t fc_n_;
  size_t fc_k_;
  size_t aligned_fc_m_;
  size_t aligned_fc_n_;
  size_t aligned_fc_k_;

  Tensor<float> *sum_per_channel_out_;
  QuantizedTensor<float, int8_t> *quantized_kernel_;
  QuantizedTensor<float, uint8_t> *quantized_data_;

  float weight_threshold_;
  float data_threshold_;
};

#endif
