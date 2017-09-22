#ifndef INTERNAL_API_H
#define INTERNAL_API_H
#include "bigquant.h"
#include <stddef.h>
#include <stdint.h>

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

extern "C" {

QuantizedConvOp *InternalQuantizedConvOpCreate();

void InternalQuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                               size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                               size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                               size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo);

void InternalQuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight);

void InternalQuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                    size_t channel_in, size_t height_in, size_t width_in);

void InternalQuantizedConvOpFree(QuantizedConvOp *p);

QuantizedFCOp *InternalQuantizedFCOpCreate();

void InternalQuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                           FC_ALGORITHM algo);

void InternalQuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight);

void InternalQuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                  size_t channel_in);

void InternalQuantizedFCOpFree(QuantizedFCOp *p);

void InternalQuantizedConvKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in,
                                         size_t kernel_h, size_t kernel_w);

void InternalQuantizedConvKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                     size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

void InternalQuantizedConvKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min,
                                              float *max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w,
                                              float threshold, LAYOUT layout);

void InternalQuantizedConvDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_in, size_t kernel_h,
                                       size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                       size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in,
                                       size_t w_in);

void InternalQuantizedConvDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_in, size_t kernel_h,
                                   size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in,
                                   float threshold, LAYOUT layout);

void InternalQuantizedConvKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out);

void InternalQuantizedConvKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w);

void InternalMixPrecisionGEMM(LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k,
                              float *ratio_a, float *ratio_b, float *kernel_sum, float *min_b, float *bias,
                              size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
                              float fault_tolerance, size_t pad_m, size_t pad_n);

void InternalQuantizedFCKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in);

void InternalQuantizedFCKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                   float threshold, LAYOUT layout);

void InternalQuantizedFCKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                            size_t c_out, size_t c_in, float threshold, LAYOUT layout);

void InternalQuantizedFCDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t batch_size, size_t channel);

void InternalQuantizedFCDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t batch_size, size_t channel,
                                 float threshold, LAYOUT layout);

void InternalQuantizedFCKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out);

void InternalQuantizedFCKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t c_out, size_t c_in);

void InternalFreeFPTensor(struct FPTensorDesc *p);

void InternalFreeQuantizedTensor(struct QuantizedTensorDesc *p);
}
#endif
