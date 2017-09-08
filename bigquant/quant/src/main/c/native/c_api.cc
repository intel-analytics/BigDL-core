#include "nn-fixpoint.h"
#include "base.h"
#include "common.h"
#include "convolution-x64.h"
#include "fc-x64.h"
#include "alloc.h"
#include "model.h"
#include "ops/ops.h"

const char* GetGitCommit() {
  char *p = GIT_VERSION;
  return p;
}

void Transpose(float *dst, float *src, size_t m, size_t n) {
  Transpose<float>(dst, src, m, n);
}

void LayoutTransform(LAYOUT dst_layout, LAYOUT src_layout, float *dst, float *src, size_t batch_size, size_t channel_size, size_t spatial_size) {
  TransformLayout<float>(dst_layout, src_layout, dst, src, batch_size, channel_size, spatial_size);
}

void FreeMemory(void *p) {
  aligned_free(p);
}

FixConvOpDesc* FixConvOpCreate(LAYOUT layout) {
  // TODO(yandai) adapt float here
  FixConvOpDesc* p = new FixConvOpDesc();
  p->layout = layout;
  if (p->layout==NCHW) {
    p->op = reinterpret_cast<FixConvOp*>(new FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>());
  } else {
    p->op = reinterpret_cast<FixConvOp*>(new FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>());
  }
  return p;
}

void FixFuseConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias,
          bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
          float *global_mean, float *global_variance, float epison, float *scale, float *shift) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupConvParameter(channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, global_variance, epison, scale, shift);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupConvParameter(channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias, conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion, global_mean, global_variance, epison, scale, shift);
  }
}

void FixConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupConvParameter(channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupConvParameter(channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias);
  }
}

void FixConvOpQuantizeKernel(FixConvOpDesc* p, float threshold) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->PrepareKernel(threshold);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->PrepareKernel(threshold);
  }
}

void FixConvOpQuantizeData(FixConvOpDesc* p, size_t batch_size, size_t channels, size_t height_in, size_t width_in, float *src, float sw_threshold) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->PrepareData(batch_size, channels, height_in, width_in, src, sw_threshold);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->PrepareData(batch_size, channels, height_in, width_in, src, sw_threshold);
  }
}

void FixConvOpExecuteToDst(FixConvOpDesc* p, float* dst, float fault_tolerance) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupTargetParameter(dst);
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->Run(fault_tolerance);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupTargetParameter(dst);
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->Run(fault_tolerance);
  }
}

void FixConvOpSetupTargetBuffer(FixConvOpDesc* p, float* dst) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupTargetParameter(dst);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupTargetParameter(dst);
  }
}

void FixConvOpExecute(FixConvOpDesc* p, float fault_tolerance) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->Run(fault_tolerance);
  } else {
    reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->Run(fault_tolerance);
  }
}


void FixConvOpFree(FixConvOpDesc* p) {
  if (p->layout == NCHW) {
    delete reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>*>(p->op);
  } else {
    delete reinterpret_cast<FixPointConvolution<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>*>(p->op);
  }
  delete p;
}

void FixConvKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w) {
  fix_tensor->dim = 2;
  fix_tensor->ori_shape[0] = c_out;
  fix_tensor->ori_shape[1] = c_in * kernel_h * kernel_w;
  fix_tensor->shape[0] = GetAlignmentLength(fix_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M);
  fix_tensor->shape[1] = GetAlignmentLength(fix_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K);
  fix_tensor->workspace_size = sizeof(int8_t) * fix_tensor->shape[0] * fix_tensor->shape[1];
  fix_tensor->workspace_size_per_meta_info = sizeof(float) * fix_tensor->shape[0];
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
}

void FixConvKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  float *tmp;
  if (layout == NHWC) {
    tmp = src;
  } else {
    aligned_malloc(reinterpret_cast<void**>(&tmp), 64, sizeof(float) * c_out * c_in * kernel_h * kernel_w);
    TransformLayout<float>(NHWC, NCHW, tmp, src, c_out, c_in, kernel_h * kernel_w);
  }
  shuffle::PadQuantizeShuffle2D<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_K>(reinterpret_cast<int8_t*>(fix_tensor->data), fix_tensor->ori_shape[0], fix_tensor->ori_shape[1],
    GetAlignmentLength(fix_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M), GetAlignmentLength(fix_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K),
    tmp, reinterpret_cast<float*>(fix_tensor->min), reinterpret_cast<float*>(fix_tensor->max), reinterpret_cast<float*>(fix_tensor->ratio), threshold);
  if (layout == NCHW) {
    aligned_free(tmp);
  }
}

void FixConvKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
  std::vector<float> fp_model(c_out * c_in * kernel_h * kernel_w);
  DequantizeModel(fp_model.data(), src, min, max, c_out, c_in, kernel_h, kernel_w);
  float *tmp;
  if (layout == NHWC) {
    tmp = fp_model.data();
  } else {
    aligned_malloc(reinterpret_cast<void**>(&tmp), 64, sizeof(float) * c_out * c_in * kernel_h * kernel_w);
    TransformLayout<float>(NHWC, NCHW, tmp, fp_model.data(), c_out, c_in, kernel_h * kernel_w);
  }
  shuffle::PadQuantizeShuffle2D<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_K>(reinterpret_cast<int8_t*>(fix_tensor->data), fix_tensor->ori_shape[0], fix_tensor->ori_shape[1],
    GetAlignmentLength(fix_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M), GetAlignmentLength(fix_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K),
    tmp, reinterpret_cast<float*>(fix_tensor->min), reinterpret_cast<float*>(fix_tensor->max), reinterpret_cast<float*>(fix_tensor->ratio), threshold);
  if (layout == NCHW) {
    aligned_free(tmp);
  }
}

void FixConvDataDescInit(FixTensor *fix_tensor, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in) {
  size_t h_out = GetConvOutSize(h_in, kernel_h, stride_h, pad_h, dilation_h);
  size_t w_out = GetConvOutSize(w_in, kernel_w, stride_w, pad_w, dilation_w);
  fix_tensor->dim = 2;
  fix_tensor->ori_shape[0] = batch_size * h_out * w_out;
  fix_tensor->ori_shape[1] = kernel_h * kernel_w * c_in;
  fix_tensor->shape[0] = GetAlignmentLength(fix_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_N);
  fix_tensor->shape[1] = GetAlignmentLength(fix_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K);
  fix_tensor->workspace_size = sizeof(uint8_t) * fix_tensor->shape[0] * fix_tensor->shape[1];
  fix_tensor->workspace_size_per_meta_info = sizeof(float) * fix_tensor->ori_shape[0];
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
}

void FixConvDataInit(FixTensor *fix_tensor, float *src, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in, float threshold, LAYOUT layout) {
  size_t h_out = GetConvOutSize(h_in, kernel_h, stride_h, pad_h, dilation_h);
  size_t w_out = GetConvOutSize(w_in, kernel_w, stride_w, pad_w, dilation_w);
  uint8_t* im2coled_data[] = {reinterpret_cast<uint8_t*>(fix_tensor->data)};
  float* im2coled_min[] = {reinterpret_cast<float*>(fix_tensor->min)};
  float* im2coled_max[] = {reinterpret_cast<float*>(fix_tensor->max)};
  float* im2coled_ratio[] = {reinterpret_cast<float*>(fix_tensor->ratio)};
  if (layout == NCHW) {
    shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(src, batch_size, c_in, 1, h_in, w_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, im2coled_data, im2coled_min, im2coled_max, im2coled_ratio, NULL, threshold, true);
  } else {
    shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(src, batch_size, c_in, 1, h_in, w_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, im2coled_data, im2coled_min, im2coled_max, im2coled_ratio, NULL, threshold, false);
  }
}

void FixConvKernelSumDescInit(FPTensor *fp_tensor, size_t c_out) {
  fp_tensor->dim = 1;
  fp_tensor->shape[0] = c_out;
  fp_tensor->workspace_size = sizeof(float) * c_out;
  aligned_malloc(&(fp_tensor->data), 64, fp_tensor->workspace_size);
}

void FixConvKernelSumInit(FPTensor *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w) {
  ComputeMatrixSumPerRow<float>(reinterpret_cast<float*>(fp_tensor->data), src, n, c * h * w);
}

void InternalMixPrecisionGEMM(LAYOUT layout, int8_t* pa, uint8_t* pb, float* pc, size_t m, size_t n, size_t k,
  float* ratio_a, float* ratio_b, float* kernel_sum, float* min_b, float* bias,
  size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
  float fault_tolerance, size_t pad_m, size_t pad_n) {
  if (layout == NCHW) {
    shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>(pa, pb, pc, m, n, k, \
      ratio_a, ratio_b, kernel_sum, min_b, bias, \
      batch_size, 1, channel_per_group, 0, height_out, width_out, \
      fault_tolerance, pad_m, pad_n);
  } else {
    shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>(pa, pb, pc, m, n, k, \
      ratio_a, ratio_b, kernel_sum, min_b, bias, \
      batch_size, 1, channel_per_group, 0, height_out, width_out, \
      fault_tolerance, pad_m, pad_n);
  }
}

FixFCOpDesc* FixFCOpCreate(LAYOUT layout) {
  // TODO(yandai) adapt float here
  FixFCOpDesc* p = new FixFCOpDesc();
  p->layout = layout;
  if (p->layout == NCHW) {
    p->op = reinterpret_cast<FixFCOp*>(new FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>());
  } else {
    p->op = reinterpret_cast<FixFCOp*>(new FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>());
  }
  return p;
}

void FixFCOpSetupFCParameter(FixFCOpDesc* p, size_t channel_out, size_t channel_in, float *src, float *bias, bool relu_fusion) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupFCParameter(channel_out, channel_in, src, bias, relu_fusion);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupFCParameter(channel_out, channel_in, src, bias, relu_fusion);
  }
}

void FixFCOpQuantizeKernel(FixFCOpDesc* p, float threshold) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->PrepareKernel(threshold);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->PrepareKernel(threshold);
  }
}

void FixFCOpQuantizeData(FixFCOpDesc* p, size_t batch_size, size_t channels, float *src, float sw_threshold) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->PrepareData(batch_size, channels, src, sw_threshold);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->PrepareData(batch_size, channels, src, sw_threshold);
  }
}

void FixFCOpExecuteToDst(FixFCOpDesc* p, float* dst, float fault_tolerance) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupTargetParameter(dst);
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->Run(fault_tolerance);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupTargetParameter(dst);
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->Run(fault_tolerance);
  }
}

void FixFCOpSetupTargetBuffer(FixFCOpDesc* p, float* dst) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->SetupTargetParameter(dst);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->SetupTargetParameter(dst);
  }
}


void FixFCOpExecute(FixFCOpDesc* p, float fault_tolerance) {
  if (p->layout == NCHW) {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op)->Run(fault_tolerance);
  } else {
    reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op)->Run(fault_tolerance);
  }
}

void FixFCOpFree(FixFCOpDesc* p) {
  if (p->layout == NCHW) {
    delete reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NCHW>*>(p->op);
  } else {
    delete reinterpret_cast<FixPointFC<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K, NHWC>*>(p->op);
  }
  delete p;
}

void FixFCKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in) {
  fix_tensor->dim = 2;
  fix_tensor->ori_shape[0] = c_out;
  fix_tensor->ori_shape[1] = c_in;
  fix_tensor->shape[0] = GetAlignmentLength(fix_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M);
  fix_tensor->shape[1] = GetAlignmentLength(fix_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K);
  fix_tensor->workspace_size = sizeof(int8_t) * fix_tensor->shape[0] * fix_tensor->shape[1];
  fix_tensor->workspace_size_per_meta_info = sizeof(float) * fix_tensor->ori_shape[0];
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
}

void FixFCKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (layout == NHWC));
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_K>(reinterpret_cast<int8_t*>(fix_tensor->data), fix_tensor->ori_shape[0], fix_tensor->ori_shape[1],
    GetAlignmentLength(fix_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M), GetAlignmentLength(fix_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K),
    src, reinterpret_cast<float*>(fix_tensor->min), reinterpret_cast<float*>(fix_tensor->max), reinterpret_cast<float*>(fix_tensor->ratio), threshold);
}

void FixFCKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (layout == NHWC));
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
  std::vector<float> fp_model(c_out * c_in);
  DequantizeModel(fp_model.data(), src, min, max, c_out, c_in, 1, 1);
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_K>(reinterpret_cast<int8_t*>(fix_tensor->data), fix_tensor->ori_shape[0], fix_tensor->ori_shape[1],
    GetAlignmentLength(fix_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M), GetAlignmentLength(fix_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K),
    fp_model.data(), reinterpret_cast<float*>(fix_tensor->min), reinterpret_cast<float*>(fix_tensor->max), reinterpret_cast<float*>(fix_tensor->ratio), threshold);
}

void FixFCDataDescInit(FixTensor *fix_tensor, size_t batch_size, size_t channel) {
  fix_tensor->dim = 2;
  fix_tensor->ori_shape[0] = batch_size;
  fix_tensor->ori_shape[1] = channel;
  fix_tensor->shape[0] = GetAlignmentLength(fix_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_N);
  fix_tensor->shape[1] = GetAlignmentLength(fix_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K);
  fix_tensor->workspace_size = sizeof(uint8_t) * fix_tensor->shape[0] * fix_tensor->shape[1];
  fix_tensor->workspace_size_per_meta_info = sizeof(float) * fix_tensor->ori_shape[0];
  aligned_malloc(&(fix_tensor->min), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->max), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->ratio), 64, fix_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(fix_tensor->data), 64, fix_tensor->workspace_size);
}

void FixFCDataInit(FixTensor *fix_tensor, float *src, size_t batch_size, size_t channel, float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (NHWC));
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K>(reinterpret_cast<uint8_t*>(fix_tensor->data), batch_size, channel,
    GetAlignmentLength(batch_size, FC_SHUFFLE_KERNEL_N), GetAlignmentLength(channel, FC_SHUFFLE_KERNEL_K),
    src, reinterpret_cast<float*>(fix_tensor->min), reinterpret_cast<float*>(fix_tensor->max), reinterpret_cast<float*>(fix_tensor->ratio), threshold);
}

void FixFCKernelSumDescInit(FPTensor *fp_tensor, size_t c_out) {
  fp_tensor->dim = 1;
  fp_tensor->shape[0] = c_out;
  fp_tensor->workspace_size = sizeof(float) * c_out;
  aligned_malloc(&(fp_tensor->data), 64, fp_tensor->workspace_size);
}

void FixFCKernelSumInit(FPTensor *fp_tensor, float *src, size_t c_out, size_t c_in) {
  ComputeMatrixSumPerRow<float>(reinterpret_cast<float*>(fp_tensor->data), src, c_out, c_in);
}
