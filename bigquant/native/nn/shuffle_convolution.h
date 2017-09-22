#ifndef NN_SHUFFLE_CONVOLUTION_H
#define NN_SHUFFLE_CONVOLUTION_H
#include "base_convolution.h"

struct ShuffleConvolutionAlgo : public BaseConvolutionAlgo {
  ShuffleConvolutionAlgo(const ConvolutionKernelDesc &conv_kernel_desc) : internal_layout_(NHWC) {
    weight_threshold_ = 64.0f;
    data_threshold_ = 127.0f;
    transformed_kernel_ = NULL;
    sum_per_channel_out_ = NULL;
    data_workspace_ = NULL;
  }

  ~ShuffleConvolutionAlgo() {
    size_t group = group_weight_.size();
    for (size_t g = 0; g < group; ++g) {
      delete group_weight_[g];
      delete quantized_weight_[g];
    }
    if (transformed_kernel_) {
      delete transformed_kernel_;
    }
    if (sum_per_channel_out_) {
      delete sum_per_channel_out_;
    }
  }

  void QuantizeKernel(float sw_threshold) {
    for (size_t g = 0; g < group_weight_.size(); ++g) {
      shuffle::PadQuantizeShuffle2D<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_K>(
          quantized_weight_[g]->data_, gemm_m_, gemm_k_, aligned_gemm_m_, aligned_gemm_k_, group_weight_[g]->data_,
          quantized_weight_[g]->min_.data_, quantized_weight_[g]->max_.data_, quantized_weight_[g]->ratio_.data_,
          sw_threshold);
    }
  }

  void KernelLayoutTransform(float *weight, const ConvolutionKernelDesc &conv_kernel_desc) {
    transformed_kernel_ =
        new Tensor<float>(make_shape(conv_kernel_desc.channel_out_, conv_kernel_desc.channel_in_per_group_,
                                     conv_kernel_desc.kernel_h_, conv_kernel_desc.kernel_w_),
                          64);
    TransformLayout(internal_layout_, conv_kernel_desc.layout_, transformed_kernel_->data_, weight,
                    conv_kernel_desc.channel_out_, conv_kernel_desc.channel_in_per_group_,
                    conv_kernel_desc.kernel_h_ * conv_kernel_desc.kernel_w_);
  }

  void ComputeKernelSum(float *weight, const ConvolutionKernelDesc &conv_kernel_desc) {
    sum_per_channel_out_ = new Tensor<float>(make_shape(conv_kernel_desc.channel_out_), 64);
    ComputeMatrixSumPerRow<float>(
        sum_per_channel_out_->data_, weight, conv_kernel_desc.channel_out_,
        conv_kernel_desc.channel_in_per_group_ * conv_kernel_desc.kernel_h_ * conv_kernel_desc.kernel_w_);
  }

  void InitWeight(float *weight, ConvolutionKernelDesc &conv_kernel_desc) {
    ComputeKernelSum(weight, conv_kernel_desc);
    if (conv_kernel_desc.layout_ != internal_layout_) {
      KernelLayoutTransform(weight, conv_kernel_desc);
      weight = transformed_kernel_->data_;
    }
    gemm_m_ = conv_kernel_desc.channel_out_per_group_;
    gemm_k_ = conv_kernel_desc.channel_in_per_group_ * conv_kernel_desc.kernel_h_ * conv_kernel_desc.kernel_w_;
    aligned_gemm_m_ = GetAlignmentLength(gemm_m_, CONV_SHUFFLE_KERNEL_M);
    aligned_gemm_k_ = GetAlignmentLength(gemm_k_, CONV_SHUFFLE_KERNEL_K);
    group_weight_.resize(conv_kernel_desc.group_);
    quantized_weight_.resize(conv_kernel_desc.group_);
    for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
      if (conv_kernel_desc.group_ != 1 && internal_layout_ == NHWC) {
        group_weight_[g] = new Tensor<float>(
            make_shape(conv_kernel_desc.channel_out_per_group_, conv_kernel_desc.channel_in_per_group_,
                       conv_kernel_desc.kernel_h_, conv_kernel_desc.kernel_w_),
            64);
      } else {
        group_weight_[g] = new Tensor<float>(make_shape(conv_kernel_desc.channel_out_per_group_,
                                                        conv_kernel_desc.channel_in_per_group_,
                                                        conv_kernel_desc.kernel_h_, conv_kernel_desc.kernel_w_));
      }
      quantized_weight_[g] = new QuantizedTensor<float, int8_t>(make_shape(aligned_gemm_m_, aligned_gemm_k_),
                                                                make_shape(conv_kernel_desc.channel_out_per_group_),
                                                                make_shape(gemm_m_, gemm_k_), 64);
    }
    if (conv_kernel_desc.group_ != 1 && internal_layout_ == NHWC) {
      std::vector<float *> group_src_ptr(conv_kernel_desc.group_);
      for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
        group_src_ptr[g] = group_weight_[g]->data_;
      }
      UnGroupKernel<float, NHWC>(group_src_ptr.data(), weight, conv_kernel_desc.group_, conv_kernel_desc.channel_out_,
                                 conv_kernel_desc.channel_in_, conv_kernel_desc.kernel_h_ * conv_kernel_desc.kernel_w_);
    } else {
      for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
        group_weight_[g]->data_ = weight +
                                  g * conv_kernel_desc.channel_out_per_group_ * conv_kernel_desc.channel_in_per_group_ *
                                      conv_kernel_desc.kernel_w_ * conv_kernel_desc.kernel_h_;
      }
    }
    QuantizeKernel(weight_threshold_);
  }

  void InitData(float *srcdata, ConvolutionDataDesc &conv_data_desc, ConvolutionKernelDesc &conv_kernel_desc,
                float sw_threshold, bool layout_transform) {
    // Allocate Memory
    height_out_ = GetConvOutSize(conv_data_desc.height_in_, conv_kernel_desc.kernel_h_, conv_kernel_desc.stride_h_,
                                 conv_kernel_desc.pad_h_, conv_kernel_desc.dilation_h_);
    width_out_ = GetConvOutSize(conv_data_desc.width_in_, conv_kernel_desc.kernel_w_, conv_kernel_desc.stride_w_,
                                conv_kernel_desc.pad_w_, conv_kernel_desc.dilation_w_);
    gemm_n_ = conv_data_desc.batch_size_ * height_out_ * width_out_;
    aligned_gemm_n_ = GetAlignmentLength(gemm_n_, CONV_SHUFFLE_KERNEL_N);
    quantized_data_.resize(conv_kernel_desc.group_);
    for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
      quantized_data_[g] = new QuantizedTensor<float, uint8_t>(make_shape(aligned_gemm_n_, aligned_gemm_k_),
                                                               make_shape(gemm_n_), make_shape(gemm_n_, gemm_k_), 64);
    }
    if (layout_transform) {
      data_workspace_ = new Tensor<float>(make_shape(conv_data_desc.batch_size_, conv_data_desc.height_in_,
                                                     conv_data_desc.width_in_, conv_data_desc.channel_in_),
                                          64);
    }
    // Init data
    std::vector<uint8_t *> quantized_data(conv_kernel_desc.group_);
    std::vector<float *> min(conv_kernel_desc.group_);
    std::vector<float *> max(conv_kernel_desc.group_);
    std::vector<float *> ratio(conv_kernel_desc.group_);
    for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
      quantized_data[g] = quantized_data_[g]->data_;
      min[g] = quantized_data_[g]->min_.data_;
      max[g] = quantized_data_[g]->max_.data_;
      ratio[g] = quantized_data_[g]->ratio_.data_;
    }
#ifdef TIME_PROFILE
    auto start = std::chrono::system_clock::now();
#endif
    if (conv_kernel_desc.layout_ == NCHW && layout_transform == false) {
      shuffle::PadQuantizeShuffleIm2colWrapper<float, NCHW>(
          srcdata, conv_data_desc.batch_size_, conv_kernel_desc.channel_in_per_group_, conv_kernel_desc.group_,
          conv_data_desc.height_in_, conv_data_desc.width_in_, conv_kernel_desc.kernel_h_, conv_kernel_desc.kernel_w_,
          conv_kernel_desc.pad_h_, conv_kernel_desc.pad_w_, conv_kernel_desc.stride_h_, conv_kernel_desc.stride_w_,
          conv_kernel_desc.dilation_h_, conv_kernel_desc.dilation_w_, quantized_data.data(), min.data(), max.data(),
          ratio.data(), data_workspace_->data_, sw_threshold, layout_transform);
    } else {
      shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(
          srcdata, conv_data_desc.batch_size_, conv_kernel_desc.channel_in_per_group_, conv_kernel_desc.group_,
          conv_data_desc.height_in_, conv_data_desc.width_in_, conv_kernel_desc.kernel_h_, conv_kernel_desc.kernel_w_,
          conv_kernel_desc.pad_h_, conv_kernel_desc.pad_w_, conv_kernel_desc.stride_h_, conv_kernel_desc.stride_w_,
          conv_kernel_desc.dilation_h_, conv_kernel_desc.dilation_w_, quantized_data.data(), min.data(), max.data(),
          ratio.data(), NULL, sw_threshold, layout_transform);
    }

#ifdef TIME_PROFILE
    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "im2col " << diff.count() << "us" << std::endl;
#endif
  }

  void Execute(float *out, float *data, float *bias, ConvolutionDataDesc &conv_data_desc,
               ConvolutionKernelDesc &conv_kernel_desc) {
    // Allocate memory
    bool transpose_data = (conv_kernel_desc.layout_ != internal_layout_) ? true : false;
    InitData(data, conv_data_desc, conv_kernel_desc, data_threshold_, transpose_data);
    // Run
    for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
#ifdef TIME_PROFILE
      auto start = std::chrono::system_clock::now();
#endif
      float *tempbias = (bias == NULL) ? bias : bias + g * conv_kernel_desc.channel_out_per_group_;
      if (conv_kernel_desc.layout_ == NCHW) {
        shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>(
            quantized_weight_[g]->data_, quantized_data_[g]->data_, out, aligned_gemm_m_, aligned_gemm_n_,
            aligned_gemm_k_, quantized_weight_[g]->ratio_.data_, quantized_data_[g]->ratio_.data_,
            sum_per_channel_out_->data_ + g * conv_kernel_desc.channel_out_per_group_, quantized_data_[g]->min_.data_,
            tempbias, conv_data_desc.batch_size_, conv_kernel_desc.group_,
            conv_kernel_desc.channel_out_ / conv_kernel_desc.group_, g, height_out_, width_out_, 0.5,
            aligned_gemm_m_ - gemm_m_, aligned_gemm_n_ - gemm_n_);
      } else {
        shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>(
            quantized_weight_[g]->data_, quantized_data_[g]->data_, out, aligned_gemm_m_, aligned_gemm_n_,
            aligned_gemm_k_, quantized_weight_[g]->ratio_.data_, quantized_data_[g]->ratio_.data_,
            sum_per_channel_out_->data_ + g * conv_kernel_desc.channel_out_per_group_, quantized_data_[g]->min_.data_,
            tempbias, conv_data_desc.batch_size_, conv_kernel_desc.group_,
            conv_kernel_desc.channel_out_ / conv_kernel_desc.group_, g, height_out_, width_out_, 0.5,
            aligned_gemm_m_ - gemm_m_, aligned_gemm_n_ - gemm_n_);
      }
#ifdef TIME_PROFILE
      auto end = std::chrono::system_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cerr << aligned_gemm_m_ << "," << aligned_gemm_n_ << "," << aligned_gemm_k_ << ",";
      std::cerr << diff.count() << "us, "
                << (2.0 * aligned_gemm_m_ * aligned_gemm_n_ * aligned_gemm_k_) / diff.count() / 1.0e3 << " glops"
                << std::endl;

#endif
    }
    FreeMemory(conv_kernel_desc);
  }

  void FreeMemory(ConvolutionKernelDesc &conv_kernel_desc) {
    for (size_t g = 0; g < conv_kernel_desc.group_; ++g) {
      delete quantized_data_[g];
    }
    if (data_workspace_) {
      delete data_workspace_;
      data_workspace_ = NULL;
    }
  }

 private:
  Tensor<float> *transformed_kernel_;
  Tensor<float> *sum_per_channel_out_;
  std::vector<Tensor<float> *> group_weight_;
  std::vector<QuantizedTensor<float, int8_t> *> quantized_weight_;
  Tensor<float> *data_workspace_;
  std::vector<QuantizedTensor<float, uint8_t> *> quantized_data_;

  const LAYOUT internal_layout_;

  size_t gemm_m_;
  size_t gemm_n_;
  size_t gemm_k_;
  size_t aligned_gemm_m_;
  size_t aligned_gemm_n_;
  size_t aligned_gemm_k_;

  float weight_threshold_;
  float data_threshold_;
};
#endif
