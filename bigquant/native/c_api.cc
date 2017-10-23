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

#include "internal_api.h"
#include "base.h"
#include "common.h"
#include "alloc.h"
#include "model.h"
#include "ops/ops.h"
#include "nn/convolution_op.h"
#include "nn/fc_op.h"

// The following is Descriptor based APU
QuantizedConvOp *InternalQuantizedConvOpCreate() {
  ConvOp *p = new ConvOp();
  return reinterpret_cast<QuantizedConvOp *>(p);
}

void InternalQuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                               size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                               size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo) {
  reinterpret_cast<ConvOp *>(p)->SetupConvolutionParameter(layout, channel_out, channel_in, group, kernel_h, kernel_w,
                                                           stride_h, stride_w, pad_h, pad_w, dialation_h, dialation_w,
                                                           fusion_mask, algo);
}

void InternalQuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight) {
  reinterpret_cast<ConvOp *>(p)->InitWeight(weight);
}

void InternalQuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                    size_t channel_in, size_t height_in, size_t width_in) {
  reinterpret_cast<ConvOp *>(p)->Execute(dst, data, bias, batch_size, channel_in, height_in, width_in);
}
void InternalQuantizedConvOpFree(QuantizedConvOp *p) {
  delete reinterpret_cast<ConvOp *>(p);
}

QuantizedFCOp *InternalQuantizedFCOpCreate() {
  FCOp *p = new FCOp();
  return reinterpret_cast<QuantizedFCOp *>(p);
}

void InternalQuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                           FC_ALGORITHM algo) {
  reinterpret_cast<FCOp *>(p)->SetupFCKernelParameter(layout, channel_out, channel_in, algo);
}

void InternalQuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight) {
  reinterpret_cast<FCOp *>(p)->InitWeight(weight);
}

void InternalQuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                  size_t channel_in) {
  reinterpret_cast<FCOp *>(p)->Execute(dst, data, bias, batch_size, channel_in);
}

void InternalQuantizedFCOpFree(QuantizedFCOp *p) {
  delete reinterpret_cast<FCOp *>(p);
}

// The following is  tensor based APU
void InternalQuantizedConvKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in,
                                         size_t kernel_h, size_t kernel_w) {
  quantized_tensor->dim = 2;
  quantized_tensor->ori_shape[0] = c_out;
  quantized_tensor->ori_shape[1] = c_in * kernel_h * kernel_w;
  quantized_tensor->shape[0] = GetAlignmentLength(quantized_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M);
  quantized_tensor->shape[1] = GetAlignmentLength(quantized_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K);
  quantized_tensor->workspace_size = sizeof(int8_t) * quantized_tensor->shape[0] * quantized_tensor->shape[1];
  quantized_tensor->workspace_size_per_meta_info = sizeof(float) * quantized_tensor->shape[0];
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
}

void InternalQuantizedConvKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                     size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  float *tmp;
  if (layout == NHWC) {
    tmp = src;
  } else {
    aligned_malloc(reinterpret_cast<void **>(&tmp), 64, sizeof(float) * c_out * c_in * kernel_h * kernel_w);
    TransformLayout<float>(NHWC, NCHW, tmp, src, c_out, c_in, kernel_h * kernel_w);
  }
  shuffle::PadQuantizeShuffle2D<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_K>(
      reinterpret_cast<int8_t *>(quantized_tensor->data), quantized_tensor->ori_shape[0],
      quantized_tensor->ori_shape[1], GetAlignmentLength(quantized_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M),
      GetAlignmentLength(quantized_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K), tmp,
      reinterpret_cast<float *>(quantized_tensor->min), reinterpret_cast<float *>(quantized_tensor->max),
      reinterpret_cast<float *>(quantized_tensor->ratio), threshold);
  if (layout == NCHW) {
    aligned_free(tmp);
  }
}

void InternalQuantizedConvKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min,
                                              float *max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w,
                                              float threshold, LAYOUT layout) {
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
  std::vector<float> fp_model(c_out * c_in * kernel_h * kernel_w);
  DequantizeModel(fp_model.data(), src, min, max, c_out, c_in, kernel_h, kernel_w);
  float *tmp;
  if (layout == NHWC) {
    tmp = fp_model.data();
  } else {
    aligned_malloc(reinterpret_cast<void **>(&tmp), 64, sizeof(float) * c_out * c_in * kernel_h * kernel_w);
    TransformLayout<float>(NHWC, NCHW, tmp, fp_model.data(), c_out, c_in, kernel_h * kernel_w);
  }
  shuffle::PadQuantizeShuffle2D<float, CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_K>(
      reinterpret_cast<int8_t *>(quantized_tensor->data), quantized_tensor->ori_shape[0],
      quantized_tensor->ori_shape[1], GetAlignmentLength(quantized_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_M),
      GetAlignmentLength(quantized_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K), tmp,
      reinterpret_cast<float *>(quantized_tensor->min), reinterpret_cast<float *>(quantized_tensor->max),
      reinterpret_cast<float *>(quantized_tensor->ratio), threshold);
  if (layout == NCHW) {
    aligned_free(tmp);
  }
}

void InternalQuantizedConvDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_in, size_t kernel_h,
                                       size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                       size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in,
                                       size_t w_in) {
  size_t h_out = GetConvOutSize(h_in, kernel_h, stride_h, pad_h, dilation_h);
  size_t w_out = GetConvOutSize(w_in, kernel_w, stride_w, pad_w, dilation_w);
  quantized_tensor->dim = 2;
  quantized_tensor->ori_shape[0] = batch_size * h_out * w_out;
  quantized_tensor->ori_shape[1] = kernel_h * kernel_w * c_in;
  quantized_tensor->shape[0] = GetAlignmentLength(quantized_tensor->ori_shape[0], CONV_SHUFFLE_KERNEL_N);
  quantized_tensor->shape[1] = GetAlignmentLength(quantized_tensor->ori_shape[1], CONV_SHUFFLE_KERNEL_K);
  quantized_tensor->workspace_size = sizeof(uint8_t) * quantized_tensor->shape[0] * quantized_tensor->shape[1];
  quantized_tensor->workspace_size_per_meta_info = sizeof(float) * quantized_tensor->ori_shape[0];
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
}

void InternalQuantizedConvDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_in, size_t kernel_h,
                                   size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in,
                                   float threshold, LAYOUT layout) {
  size_t h_out = GetConvOutSize(h_in, kernel_h, stride_h, pad_h, dilation_h);
  size_t w_out = GetConvOutSize(w_in, kernel_w, stride_w, pad_w, dilation_w);
  uint8_t *im2coled_data[] = {reinterpret_cast<uint8_t *>(quantized_tensor->data)};
  float *im2coled_min[] = {reinterpret_cast<float *>(quantized_tensor->min)};
  float *im2coled_max[] = {reinterpret_cast<float *>(quantized_tensor->max)};
  float *im2coled_ratio[] = {reinterpret_cast<float *>(quantized_tensor->ratio)};
  if (layout == NCHW) {
    shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(
        src, batch_size, c_in, 1, h_in, w_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, im2coled_data, im2coled_min, im2coled_max, im2coled_ratio, NULL, threshold, true);
  } else {
    shuffle::PadQuantizeShuffleIm2colWrapper<float, NHWC>(
        src, batch_size, c_in, 1, h_in, w_in, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, im2coled_data, im2coled_min, im2coled_max, im2coled_ratio, NULL, threshold, false);
  }
}

void InternalQuantizedConvKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out) {
  fp_tensor->dim = 1;
  fp_tensor->shape[0] = c_out;
  fp_tensor->workspace_size = sizeof(float) * c_out;
  aligned_malloc(&(fp_tensor->data), 64, fp_tensor->workspace_size);
}

void InternalQuantizedConvKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w) {
  ComputeMatrixSumPerRow<float>(reinterpret_cast<float *>(fp_tensor->data), src, n, c * h * w);
}

void InternalMixPrecisionGEMM(LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k,
                              float *ratio_a, float *ratio_b, float *kernel_sum, float *min_b, float *bias,
                              size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
                              float fault_tolerance, size_t pad_m, size_t pad_n) {
  if (layout == NCHW) {
    shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NCHW>(
        pa, pb, pc, m, n, k, ratio_a, ratio_b, kernel_sum, min_b, bias, batch_size, 1, channel_per_group, 0, height_out,
        width_out, fault_tolerance, pad_m, pad_n);
  } else {
    shuffle::ConvShuffleGEMM<CONV_SHUFFLE_KERNEL_M, CONV_SHUFFLE_KERNEL_N, CONV_SHUFFLE_KERNEL_K, NHWC>(
        pa, pb, pc, m, n, k, ratio_a, ratio_b, kernel_sum, min_b, bias, batch_size, 1, channel_per_group, 0, height_out,
        width_out, fault_tolerance, pad_m, pad_n);
  }
}

void InternalQuantizedFCKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in) {
  quantized_tensor->dim = 2;
  quantized_tensor->ori_shape[0] = c_out;
  quantized_tensor->ori_shape[1] = c_in;
  quantized_tensor->shape[0] = GetAlignmentLength(quantized_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M);
  quantized_tensor->shape[1] = GetAlignmentLength(quantized_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K);
  quantized_tensor->workspace_size = sizeof(int8_t) * quantized_tensor->shape[0] * quantized_tensor->shape[1];
  quantized_tensor->workspace_size_per_meta_info = sizeof(float) * quantized_tensor->ori_shape[0];
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
}

void InternalQuantizedFCKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                   float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (layout == NHWC));
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_K>(
      reinterpret_cast<int8_t *>(quantized_tensor->data), quantized_tensor->ori_shape[0],
      quantized_tensor->ori_shape[1], GetAlignmentLength(quantized_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M),
      GetAlignmentLength(quantized_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K), src,
      reinterpret_cast<float *>(quantized_tensor->min), reinterpret_cast<float *>(quantized_tensor->max),
      reinterpret_cast<float *>(quantized_tensor->ratio), threshold);
}

void InternalQuantizedFCKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                            size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (layout == NHWC));
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
  std::vector<float> fp_model(c_out * c_in);
  DequantizeModel(fp_model.data(), src, min, max, c_out, c_in, 1, 1);
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_M, FC_SHUFFLE_KERNEL_K>(
      reinterpret_cast<int8_t *>(quantized_tensor->data), quantized_tensor->ori_shape[0],
      quantized_tensor->ori_shape[1], GetAlignmentLength(quantized_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_M),
      GetAlignmentLength(quantized_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K), fp_model.data(),
      reinterpret_cast<float *>(quantized_tensor->min), reinterpret_cast<float *>(quantized_tensor->max),
      reinterpret_cast<float *>(quantized_tensor->ratio), threshold);
}

void InternalQuantizedFCDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t batch_size, size_t channel) {
  quantized_tensor->dim = 2;
  quantized_tensor->ori_shape[0] = batch_size;
  quantized_tensor->ori_shape[1] = channel;
  quantized_tensor->shape[0] = GetAlignmentLength(quantized_tensor->ori_shape[0], FC_SHUFFLE_KERNEL_N);
  quantized_tensor->shape[1] = GetAlignmentLength(quantized_tensor->ori_shape[1], FC_SHUFFLE_KERNEL_K);
  quantized_tensor->workspace_size = sizeof(uint8_t) * quantized_tensor->shape[0] * quantized_tensor->shape[1];
  quantized_tensor->workspace_size_per_meta_info = sizeof(float) * quantized_tensor->ori_shape[0];
  aligned_malloc(&(quantized_tensor->min), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->max), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->ratio), 64, quantized_tensor->workspace_size_per_meta_info);
  aligned_malloc(&(quantized_tensor->data), 64, quantized_tensor->workspace_size);
}

void InternalQuantizedFCDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t batch_size, size_t channel,
                                 float threshold, LAYOUT layout) {
  assert((layout == NCHW) || (layout == NHWC));
  shuffle::PadQuantizeShuffle2D<float, FC_SHUFFLE_KERNEL_N, FC_SHUFFLE_KERNEL_K>(
      reinterpret_cast<uint8_t *>(quantized_tensor->data), batch_size, channel,
      GetAlignmentLength(batch_size, FC_SHUFFLE_KERNEL_N), GetAlignmentLength(channel, FC_SHUFFLE_KERNEL_K), src,
      reinterpret_cast<float *>(quantized_tensor->min), reinterpret_cast<float *>(quantized_tensor->max),
      reinterpret_cast<float *>(quantized_tensor->ratio), threshold);
}

void InternalQuantizedFCKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out) {
  fp_tensor->dim = 1;
  fp_tensor->shape[0] = c_out;
  fp_tensor->workspace_size = sizeof(float) * c_out;
  aligned_malloc(&(fp_tensor->data), 64, fp_tensor->workspace_size);
}

void InternalQuantizedFCKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t c_out, size_t c_in) {
  ComputeMatrixSumPerRow<float>(reinterpret_cast<float *>(fp_tensor->data), src, c_out, c_in);
}

void InternalFreeFPTensor(struct FPTensorDesc *p) {
  aligned_free(p->data);
}

void InternalFreeQuantizedTensor(struct QuantizedTensorDesc *p) {
  aligned_free(p->data);
  aligned_free(p->min);
  aligned_free(p->max);
  aligned_free(p->ratio);
}
