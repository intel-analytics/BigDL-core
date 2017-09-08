#ifndef FC_X64
#define FC_X64

#include "base.h"
#include "common.h"
#include "tensor.h"
#include "ops/ops.h"

#ifdef TIME_PROFILE
#include <chrono>
#endif


template <typename SrcType, typename DstType, size_t shuffle_rows, size_t shuffle_cols, LAYOUT layout>
struct FCKernel {
  FCKernel(size_t channel_out, size_t channel_in, SrcType *src, SrcType *bias):
    channel_out_(channel_out),
    channel_in_(channel_in),
    src_(src),
    bias_(bias) {
  }

  FCKernel() {

  }

  ~FCKernel() {
    delete sumperrow_;
    delete dst_;
  }

  void SetupParameter(size_t channel_out, size_t channel_in, SrcType *src, SrcType *bias) {
    channel_out_ = channel_out;
    channel_in_ = channel_in;
    src_ = src;
    bias_ = bias;
  }

  void Init(float sw_threshold = 127.0f) {
    GetPadSizeForFixPointGemm();
    sumperrow_ = new Tensor<SrcType>(make_shape(channel_out_), 64);
    dst_ = new QuantizedTensor<SrcType, DstType>(make_shape(m_, n_), make_shape(valid_m_), make_shape(valid_m_, valid_n_), 64);
    ComputeMatrixSumPerRow<SrcType>(sumperrow_->data_, src_, channel_out_, channel_in_);
    QuantizeKernel(sw_threshold);
  }

  void QuantizeKernel(float sw_threshold) {
    aligned_malloc(reinterpret_cast<void**>(&dst_->data_), 64, sizeof(DstType) * m_ * n_);
    shuffle::PadQuantizeShuffle2D<SrcType, shuffle_rows, shuffle_cols>(dst_->data_, valid_m_, valid_n_, m_, n_, src_, dst_->min_.data_, dst_->max_.data_, dst_->ratio_.data_, sw_threshold);
  }

  void GetPadSizeForFixPointGemm() {
    valid_m_ = channel_out_;
    valid_n_ = channel_in_;
    m_ = GetAlignmentLength(valid_m_, shuffle_rows);
    n_ = GetAlignmentLength(valid_n_, shuffle_cols);
    pad_m_ = m_ - valid_m_;
    pad_n_ = n_ - valid_n_;
  }

  size_t channel_out_;
  size_t channel_in_;

  SrcType *src_;
  SrcType *bias_;

  Tensor<SrcType> *sumperrow_;
  QuantizedTensor<SrcType, DstType>* dst_;

  size_t m_;
  size_t n_;
  size_t pad_m_;
  size_t pad_n_;
  size_t valid_m_;
  size_t valid_n_;
};

template <typename SrcType, typename DstType, size_t shuffle_rows, size_t shuffle_cols, LAYOUT layout>
struct FCData {
  FCData(size_t batch_size, size_t channels, SrcType *src):
    batch_size_(batch_size),
    channels_(channels),
    layout_(layout),
    src_(src) {
  }

  FCData() {

  }

  ~FCData() {
    //aligned_free(dst_);
  }

  void SetupParameter(size_t batch_size, size_t channels, SrcType *src) {
    batch_size_ = batch_size;
    channels_ = channels;
    layout_ = layout;
    src_ = src;
  }

  void Init(float sw_threshold = 255.0f) {
    GetPadSizeForFixPointGemm();
    dst_ = new QuantizedTensor<SrcType, DstType>(make_shape(m_, n_), make_shape(valid_m_), make_shape(valid_m_, valid_n_), 64);;
#ifdef TIME_PROFILE
    auto start = std::chrono::system_clock::now();
#endif
    shuffle::PadQuantizeShuffle2D<SrcType, shuffle_rows, shuffle_cols>(dst_->data_, valid_m_, valid_n_, m_, n_, src_, dst_->min_.data_, dst_->max_.data_, dst_->ratio_.data_, sw_threshold);
#ifdef TIME_PROFILE
    auto end = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cerr << "pad shuffle fc input " << diff.count() << "us" << std::endl;
#endif
  }


  void FreeMemory() {
    delete dst_;
    dst_ = NULL;
  }

  void GetPadSizeForFixPointGemm() {
    valid_m_ = batch_size_;
    valid_n_ = channels_;
    m_ = GetAlignmentLength(valid_m_, shuffle_rows);
    n_ = GetAlignmentLength(valid_n_, shuffle_cols);
    pad_m_ = m_ - valid_m_;
    pad_n_ = n_ - valid_n_;
  }


  size_t batch_size_;
  size_t channels_;
  size_t group_;
  size_t channel_per_group_;

  size_t m_;
  size_t n_;
  size_t pad_m_;
  size_t pad_n_;
  size_t valid_m_;
  size_t valid_n_;


  LAYOUT layout_;
  SrcType *src_;
  QuantizedTensor<SrcType, DstType>* dst_;
};

template <typename DType, size_t weight_shuffle_rows, size_t data_shuffle_rows, size_t shuffle_cols, LAYOUT layout>
struct FixPointFC {
  FixPointFC() : fcdata_(), fckernel_() {

  }

  void SetupFCParameter(size_t channel_out, size_t channel_in, DType *src, DType *bias, bool relu_fusion = false) {
    channel_out_ = channel_out;
    channel_in_ = channel_in;
    layout_ = layout;
    kernelsrc_ = src;
    bias_ = bias;
    relu_fusion_ = relu_fusion;
  }

  void PrepareData(size_t batch_size, size_t channels, DType *src, float sw_threshold = 255.0f) {
    batch_size_ = batch_size;
    data_channels_ = channels;
    datasrc_ = src;
    assert(layout == layout_);
    fcdata_.SetupParameter(batch_size_, data_channels_, datasrc_);
    fcdata_.Init(sw_threshold);
  }

  void SetupTargetParameter(DType *dst) {
    dst_ = dst;
  }

  void PrepareKernel(float sw_threshold = 127.0f) {
    fckernel_.SetupParameter(channel_out_, channel_in_, kernelsrc_, bias_);
    fckernel_.Init(sw_threshold);
  }


  void Run(float fault_tolerance = 0.5) {
#ifdef TIME_PROFILE
      auto start = std::chrono::system_clock::now();
#endif
      shuffle::ConvShuffleGEMM<weight_shuffle_rows, data_shuffle_rows, shuffle_cols, layout>(fckernel_.dst_->data_, fcdata_.dst_->data_, dst_, fckernel_.m_, fcdata_.m_, fckernel_.n_, fckernel_.dst_->ratio_.data_, fcdata_.dst_->ratio_.data_, fckernel_.sumperrow_->data_, fcdata_.dst_->min_.data_, bias_, batch_size_, 1, channel_out_, 0, 1, 1, fault_tolerance, fckernel_.pad_m_, fcdata_.pad_m_, relu_fusion_);
#ifdef TIME_PROFILE
      auto end = std::chrono::system_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cerr << fckernel_.m_ << "," << fcdata_.m_ << "," << fckernel_.n_ << ",";
      std::cerr << diff.count() << "us, " << (2.0 * fckernel_.m_ * fcdata_.m_ * fckernel_.n_) / diff.count() / 1.0e3 << " glops" <<std::endl;
#endif
    fcdata_.FreeMemory();
  }

  // kernel parameters
  struct FCData<DType, uint8_t, data_shuffle_rows, shuffle_cols, layout> fcdata_;
  struct FCKernel<DType, int8_t, weight_shuffle_rows, shuffle_cols, layout> fckernel_;
  size_t channel_out_;
  size_t channel_in_;
  LAYOUT layout_;
  DType *kernelsrc_;
  DType *bias_;

  // data parameters
  size_t batch_size_;
  size_t data_channels_;
  DType *datasrc_;

  DType *dst_;

  // op fusion
  bool relu_fusion_;
};
#endif
