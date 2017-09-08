#ifndef CONVOLUTION_AVX2
#define CONVOLUTION_AVX2

#include "base.h"
#include "common.h"
#include "tensor.h"
#include "ops/ops.h"
#ifdef TIME_PROFILE
#include <chrono>
#endif


template <typename SrcType, typename DstType, size_t shuffle_rows, size_t shuffle_cols>
struct ConvKernel {
  ConvKernel(size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, SrcType *src, SrcType *bias) :
    channel_out_(channel_out),
    channel_in_(channel_in),
    group_(group),
    channel_out_per_group_(channel_out / group_),
    channel_in_per_group_(channel_in / group_),
    kernel_h_(kernel_h),
    kernel_w_(kernel_w),
    stride_h_(stride_h),
    stride_w_(stride_w),
    pad_h_(pad_h),
    pad_w_(pad_w),
    src_(src),
    bias_(bias) {

  }

  ConvKernel(LAYOUT layout) {
    layout_ = layout;
  }

  ConvKernel() {

  }

  ~ConvKernel() {
    // free group memory
    for (size_t g = 0; g < group_; ++g) {
      delete group_src_[g];
    }
    // free sumperrpw
    delete sumperrow_;
    // free quantized tensor
    for (size_t g = 0; g < group_; ++g) {
      delete dst_[g];
    }
  }

  void SetLayout(LAYOUT layout) {
    layout_ = layout;
  }

  void SetupParameter(size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, SrcType *src, SrcType *bias) {
    channel_out_ = channel_out;
    channel_in_ = channel_in;
    group_ = group;
    channel_out_per_group_ = channel_out / group_;
    channel_in_per_group_ = channel_in / group_;
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
    src_ = src;
    bias_ = bias;
  }

  void Allocate_Memory() {
    GetFixpointKernelShape();
    group_src_.resize(group_);
    dst_.resize(group_);
    sumperrow_ = new Tensor<SrcType>(make_shape(channel_out_), 64);
    Shape s = make_shape(channel_out_per_group_);
    for (size_t g = 0; g < group_; ++g) {
      if (group_ != 1 && layout_ == NHWC) {
        group_src_[g] = new Tensor<SrcType>(make_shape(channel_out_per_group_ ,channel_in_per_group_, kernel_h_, kernel_w_), 64);
      } else {
        for (size_t g = 0; g < group_; ++g) {
          group_src_[g] = new Tensor<SrcType>(make_shape(channel_out_per_group_ ,channel_in_per_group_, kernel_h_, kernel_w_));
        }
      }
      dst_[g] = new QuantizedTensor<SrcType, DstType>(make_shape(m_, n_), s, make_shape(valid_m_, valid_n_), 64);
    }
  }

  void Init(float sw_threshold = 127.0f) {
    Allocate_Memory();
    ComputeMatrixSumPerRow<SrcType>(sumperrow_->data_, src_, channel_out_, channel_in_ * kernel_h_ * kernel_w_);
    std::vector<SrcType*> group_src_ptr(group_);
    if (group_ != 1 && layout_ == NHWC) {
      for (size_t g = 0; g < group_; ++g) {
        group_src_ptr[g] = group_src_[g]->data_;
      }
      UnGroupKernel<SrcType, NHWC>(group_src_ptr.data(), src_, group_, channel_out_, channel_in_, kernel_h_ * kernel_w_);
    } else {
      for (size_t g = 0; g < group_; ++g) {
        group_src_[g]->data_ = src_ + g * channel_out_per_group_ * channel_in_per_group_ * kernel_w_ * kernel_h_;
      }
    }
    QuantizeKernel(sw_threshold);
  }

  void QuantizeKernel(float sw_threshold) {
    for (size_t g = 0; g < group_; ++g) {
      shuffle::PadQuantizeShuffle2D<SrcType, shuffle_rows, shuffle_cols>(dst_[g]->data_, valid_m_, valid_n_, m_, n_, group_src_[g]->data_, dst_[g]->min_.data_, dst_[g]->max_.data_, dst_[g]->ratio_.data_, sw_threshold);
    }
  }

  void GetFixpointKernelShape() {
    valid_m_ = channel_out_per_group_;
    valid_n_ = channel_in_per_group_ * kernel_h_ * kernel_w_;
    m_ = GetAlignmentLength(valid_m_, shuffle_rows);
    n_ = GetAlignmentLength(valid_n_, shuffle_cols);
    pad_m_ = m_ - valid_m_;
    pad_n_ = n_ - valid_n_;
  }


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

  LAYOUT layout_;

  SrcType *src_;
  SrcType *bias_;


  size_t valid_m_;
  size_t valid_n_;
  size_t m_;
  size_t n_;
  size_t pad_m_;
  size_t pad_n_;

  Tensor<SrcType> *sumperrow_;
  std::vector<Tensor<SrcType>*> group_src_;
  std::vector<QuantizedTensor<SrcType, DstType>*> dst_;
};

template <typename SrcType, typename DstType, size_t shuffle_rows, size_t shuffle_cols>
struct ConvData {
  ConvData(size_t batch_size, size_t channels, size_t group, size_t height_in, size_t width_in, SrcType *src):
    batch_size_(batch_size),
    channels_(channels),
    group_(group),
    channel_per_group_(channels_ / group_),
    height_in_(height_in),
    width_in_(width_in),
    src_(src) {
  }

  ConvData() : workspace_(NULL) {

  }

  void SetLayout(LAYOUT layout) {
    layout_ = layout;
  }

  void SetupParameter(size_t batch_size, size_t channels, size_t group, size_t height_in, size_t width_in, SrcType *src) {
    batch_size_ = batch_size;
    channels_ = channels;
    group_ = group;
    channel_per_group_ = channels_ / group_;
    height_in_ = height_in;
    width_in_ = width_in;
    src_ = src;
  }

  ~ConvData() {

  }

  void Allocate_Memory(size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h, size_t pad_w, bool transposed) {
    height_out_ = GetConvOutSize(height_in_, kernel_h, stride_h, pad_h, dilation_h);
    width_out_ = GetConvOutSize(width_in_, kernel_w, stride_w, pad_w, dilation_w);
    size_t total_cols = batch_size_ * height_out_ * width_out_;
    GetFixpointDataShape(kernel_h, kernel_w);
    dst_.resize(group_);
    for (size_t g = 0; g < group_; ++g) {
      dst_[g] = new QuantizedTensor<SrcType, DstType>(make_shape(m_, n_), make_shape(total_cols), make_shape(valid_m_, valid_n_), 64);
    }
    if (transposed) {
      workspace_ = new Tensor<SrcType>(make_shape(batch_size_, height_in_, width_in_, channels_), 64);
    }
  }

  void Init(size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h, size_t pad_w, float sw_threshold = 255.0f, bool transposed = false) {
    Allocate_Memory(kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, transposed);
    std::vector<DstType*> dst(group_);
    std::vector<SrcType*> min(group_);
    std::vector<SrcType*> max(group_);
    std::vector<SrcType*> ratio(group_);
    for (size_t g = 0; g < group_; ++g) {
      dst[g] = dst_[g]->data_;
      min[g] = dst_[g]->min_.data_;
      max[g] = dst_[g]->max_.data_;
      ratio[g] = dst_[g]->ratio_.data_;
    }
#ifdef TIME_PROFILE
    auto start = std::chrono::system_clock::now();
#endif
    if (layout_ == NCHW) {
      shuffle::PadQuantizeShuffleIm2colWrapper<SrcType, NCHW>(src_, batch_size_, channel_per_group_, group_, height_in_, width_in_,
                                          kernel_h, kernel_w,
                                          pad_h, pad_w,
                                          stride_h, stride_w,
                                          dilation_h, dilation_w,
                                          dst.data(), min.data(), max.data(), ratio.data(), workspace_->data_, sw_threshold, transposed);
    } else {
      shuffle::PadQuantizeShuffleIm2colWrapper<SrcType, NHWC>(src_, batch_size_, channel_per_group_, group_, height_in_, width_in_,
                                          kernel_h, kernel_w,
                                          pad_h, pad_w,
                                          stride_h, stride_w,
                                          dilation_h, dilation_w,
                                          dst.data(), min.data(), max.data(), ratio.data(), workspace_->data_, sw_threshold, transposed);
    }
#ifdef TIME_PROFILE
    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "im2col " << diff.count() << "us" << std::endl;
#endif
  }

  void FreeMemory() {
    for (size_t g = 0; g < group_; ++g) {
      delete dst_[g];
    }
    if (workspace_ != NULL) {
      delete workspace_;
    }
  }

  void GetFixpointDataShape(size_t kernel_h, size_t kernel_w) {
    valid_m_ = batch_size_ * height_out_ * width_out_;
    valid_n_ = kernel_h * kernel_w * channels_ / group_;
    m_ = GetAlignmentLength(valid_m_, shuffle_rows);
    n_ = GetAlignmentLength(valid_n_, shuffle_cols);
    pad_m_ = m_ - valid_m_;
    pad_n_ = n_ - valid_n_;
  }


  size_t batch_size_;
  size_t channels_;
  size_t group_;
  size_t channel_per_group_;
  size_t height_in_;
  size_t width_in_;
  size_t height_out_;
  size_t width_out_;

  size_t pad_m_;
  size_t pad_n_;
  size_t m_;
  size_t n_;
  size_t valid_m_;
  size_t valid_n_;


  LAYOUT layout_;
  SrcType *src_;
  Tensor<SrcType>* workspace_;
  std::vector<QuantizedTensor<SrcType, DstType>*> dst_;
};

template <typename DType, size_t weight_shuffle_rows, size_t data_shuffle_rows, size_t shuffle_cols, LAYOUT layout>
struct FixPointConvolution {
  FixPointConvolution() : convkernel_(), enable_internal_layout_(true), transformed_kernel_(NULL), transformed_data_(NULL), mul_variance_coeff_(NULL) {
    // if internal layout is not the same as
    if (layout == NCHW && enable_internal_layout_ == true) {
      internal_layout_ = NHWC;
    } else {
      internal_layout_ = layout;
    }
    convkernel_.SetLayout(internal_layout_);
    convdata_.SetLayout(internal_layout_);
  }

  ~FixPointConvolution() {
    if (transformed_kernel_) {
      free(transformed_kernel_);
    }
    if (conv_bn_fusion_ || conv_relu_bn_fusion_ || conv_bn_relu_fusion_) {
      delete mul_variance_coeff_;
    }
  }


  void SetupConvParameter(size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h, size_t pad_w, DType *src, DType *bias,
            bool conv_relu_fusion = false, bool conv_bn_fusion = false, bool conv_bn_relu_fusion = false, bool conv_relu_bn_fusion = false,
            DType *global_mean = NULL, DType *global_variance = NULL, DType epison = 0.0f, DType *scale = NULL, DType *shift = NULL) {
    channel_out_ = channel_out;
    channel_in_ = channel_in;
    group_ = group;
    channel_out_per_group_ = channel_out_ / group_;
    channel_in_per_group_ = channel_in_ / group_;
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    dilation_h_ = dilation_h;
    dilation_w_ = dilation_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
    layout_ = layout;
    kernelsrc_ = src;
    bias_ = bias;
    bool fusion_cond0 = (conv_relu_fusion == false) && (conv_bn_fusion == false) && (conv_bn_relu_fusion == false) && (conv_relu_bn_fusion == false);
    bool fusion_cond1 = (conv_relu_fusion == true) && (conv_bn_fusion == false) && (conv_bn_relu_fusion == false) && (conv_relu_bn_fusion == false);
    bool fusion_cond2 = (conv_relu_fusion == false) && (conv_bn_fusion == true) && (conv_bn_relu_fusion == false) && (conv_relu_bn_fusion == false);
    bool fusion_cond3 = (conv_relu_fusion == false) && (conv_bn_fusion == false) && (conv_bn_relu_fusion == true) && (conv_relu_bn_fusion == false);
    bool fusion_cond4 = (conv_relu_fusion == false) && (conv_bn_fusion == false) && (conv_bn_relu_fusion == false) && (conv_relu_bn_fusion == true);
    assert(fusion_cond0 || fusion_cond1 || fusion_cond2 || fusion_cond3 || fusion_cond4);
    conv_relu_fusion_ = conv_relu_fusion;
    conv_bn_fusion_ = conv_bn_fusion;
    conv_bn_relu_fusion_ = conv_bn_relu_fusion;
    conv_relu_bn_fusion_ = conv_relu_bn_fusion;
    global_mean_ = global_mean;
    global_variance_ = global_variance;
    epison_ = epison;
    scale_ = scale;
    shift_ = shift;
  }

  void PrepareKernel(float sw_threshold = 127.0f) {
    // TODO(yandai) change this to tensor API
    if (layout != internal_layout_) {
      aligned_malloc(reinterpret_cast<void**>(&transformed_kernel_), 64, channel_out_ * channel_in_ * kernel_h_ * kernel_w_ * sizeof(DType));
      TransformLayout(internal_layout_, layout, transformed_kernel_, kernelsrc_, channel_out_, channel_in_per_group_, kernel_h_ * kernel_w_);
      kernelsrc_ = transformed_kernel_;
    }
    if (conv_bn_fusion_ || conv_relu_bn_fusion_ || conv_bn_relu_fusion_) {
      mul_variance_coeff_ = new Tensor<DType>(make_shape(channel_out_), 64);
      for (size_t i = 0; i < channel_out_; ++i) {
        mul_variance_coeff_->data_[i] = 1.0 / sqrtf(global_variance_[i] + epison_);
      }
    }
    convkernel_.SetupParameter(channel_out_, channel_in_, group_, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, kernelsrc_, bias_);
    convkernel_.Init(sw_threshold);
  }

  void PrepareData(size_t batch_size, size_t channels, size_t height_in, size_t width_in, DType *src, float sw_threshold = 255.0f, bool transpose = false) {
    if (layout != internal_layout_) {
      transpose = true;
    } else {
      transpose = false;
    }
    datasrc_ = src;
    batch_size_ = batch_size;
    data_channels_ = channels;
    height_in_ = height_in;
    width_in_ = width_in;
    convdata_.SetupParameter(batch_size_, data_channels_, group_, height_in_, width_in, datasrc_);
    convdata_.Init(kernel_h_, kernel_w_, stride_h_, stride_w_, dilation_h_, dilation_w_, pad_h_, pad_w_, sw_threshold, transpose);
  }

  void SetupTargetParameter(DType *dst) {
    dst_ = dst;
  }


  // The following two APIs are very dangerous. Users should avoid using them.
  std::vector<QuantizedTensor<DType, int8_t>*>& GetQuantizedKernelTensor() {
    return convkernel_.dst_;
  }

  std::vector<QuantizedTensor<DType, uint8_t>*> GetQuantizedDataTensor() {
    return convdata_.dst_;
  }

  Tensor<DType>* GetKernelRowSum() {
    return convkernel_.sumperrow_;
  }

  void Run(float fault_tolerance = 0.5) {
    for (size_t g = 0; g < group_; ++g) {
#ifdef TIME_PROFILE
      auto start = std::chrono::system_clock::now();
#endif
      DType *p_mul_variance = (mul_variance_coeff_ == NULL)? NULL : mul_variance_coeff_->data_;
      shuffle::ConvShuffleGEMM<weight_shuffle_rows, data_shuffle_rows, shuffle_cols, layout>(convkernel_.dst_[g]->data_, convdata_.dst_[g]->data_, dst_, convkernel_.m_, convdata_.m_, convkernel_.n_, convkernel_.dst_[g]->ratio_.data_, convdata_.dst_[g]->ratio_.data_, convkernel_.sumperrow_->data_ + g * channel_out_per_group_, convdata_.dst_[g]->min_.data_, convkernel_.bias_ + g * channel_out_per_group_, batch_size_, group_, channel_out_ / group_, g, convdata_.height_out_, convdata_.width_out_, fault_tolerance, convkernel_.pad_m_, convdata_.pad_m_, conv_relu_fusion_, conv_bn_fusion_, conv_bn_relu_fusion_, conv_relu_bn_fusion_, global_mean_, p_mul_variance, scale_, shift_);
#ifdef TIME_PROFILE
      auto end = std::chrono::system_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cerr << convkernel_.m_ << "," << convdata_.m_ << "," << convkernel_.n_ << ",";
      std::cerr << diff.count() << "us, " << (2.0 * convkernel_.m_ * convdata_.m_ * convkernel_.n_) / diff.count() / 1.0e3 << " glops" <<std::endl;
#endif
    }
    convdata_.FreeMemory();
    if (transformed_data_) {
      free(transformed_data_);
    }
  }

  // kernel parameters
  struct ConvData<DType, uint8_t, data_shuffle_rows, shuffle_cols> convdata_;
  struct ConvKernel<DType, int8_t, weight_shuffle_rows, shuffle_cols> convkernel_;
  size_t channel_out_;
  size_t channel_in_;
  size_t group_;
  size_t channel_out_per_group_;
  size_t channel_in_per_group_;
  size_t dilation_h_;
  size_t dilation_w_;
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  LAYOUT layout_;
  LAYOUT internal_layout_;
  DType *kernelsrc_;
  DType *transformed_kernel_;
  DType *transformed_data_;
  DType *bias_;

  // data parameters
  size_t batch_size_;
  size_t data_channels_;
  size_t height_in_ ;
  size_t width_in_;
  DType *datasrc_;

  DType *dst_;

  // op fusion
  // conv + relu
  bool conv_relu_fusion_;
  // conv + bn
  bool conv_bn_fusion_;
  // conv + bn + relu
  bool conv_bn_relu_fusion_;
  // conv + relu + bn
  bool conv_relu_bn_fusion_;
  DType *global_mean_;
  DType *global_variance_;
  DType epison_;
  DType *scale_;
  DType *shift_;
  Tensor<DType> *mul_variance_coeff_;
  bool enable_internal_layout_;
};
#endif
