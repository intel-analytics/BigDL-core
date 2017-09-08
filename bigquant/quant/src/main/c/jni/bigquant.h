#ifndef BIGQUANT_H
#define BIGQUANT_H
#include <stddef.h>
#include <stdint.h>

typedef enum LAYOUT { NCHW = 0, NHWC = 1 } LAYOUT;

struct FPTensorDesc {
  void *data;
  size_t shape[4];
  size_t dim;
  size_t workspace_size;
};
typedef struct FPTensorDesc FPTensor;

struct QuantizedTensorDesc {
  void *data;
  void *min;
  void *max;
  void *ratio;
  size_t shape[4];
  size_t ori_shape[4];
  size_t dim;
  size_t workspace_size_per_meta_info;
  size_t workspace_size;
};

typedef struct QuantizedTensorDesc QuantizedTensor;

struct QuantizedConvOp;
typedef struct QuantizedConvOp QuantizedConvOp;

struct QuantizedConvOpDesc {
  LAYOUT layout;
  QuantizedConvOp *op;
};
typedef struct QuantizedConvOpDesc QuantizedConvOpDesc;

struct QuantizedFCOp;
typedef struct QuantizedFCOp QuantizedFCOp;

struct QuantizedFCOpDesc {
  LAYOUT layout;
  QuantizedFCOp *op;
};
typedef struct QuantizedFCOpDesc QuantizedFCOpDesc;

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif
API_PREFIX void ManualRuntimeLoadLib(const char *path);

API_PREFIX void FreeMemory(void *p);

API_PREFIX const char *GetGitCommit();

API_PREFIX void Transpose(float *dst, float *src, size_t m, size_t n);

API_PREFIX void LayoutTransform(LAYOUT dst_layout, LAYOUT src_layout, float *dst, float *src, size_t batch_size,
                                size_t channel_size, size_t spatial_size);

API_PREFIX QuantizedConvOpDesc *QuantizedConvOpCreate(LAYOUT layout);

API_PREFIX void QuantizedConvOpSetupConvParameter(QuantizedConvOpDesc *p, size_t channel_out, size_t channel_in,
                                                  size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                                  size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h,
                                                  size_t pad_w, float *src, float *bias);

API_PREFIX void QuantizedFuseConvOpSetupConvParameter(
    QuantizedConvOpDesc *p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src,
    float *bias, bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
    float *global_mean, float *global_variance, float epison, float *scale, float *shift);

API_PREFIX void QuantizedConvOpQuantizeKernel(QuantizedConvOpDesc *p, float threshold);

API_PREFIX void QuantizedConvOpQuantizeData(QuantizedConvOpDesc *p, size_t batch_size, size_t channels,
                                            size_t height_in, size_t width_in, float *src, float sw_threshold);

API_PREFIX void QuantizedConvOpExecuteToDst(QuantizedConvOpDesc *p, float *dst, float fault_tolerance);

API_PREFIX void QuantizedConvOpSetupTargetBuffer(QuantizedConvOpDesc *p, float *dst);

API_PREFIX void QuantizedConvOpExecute(QuantizedConvOpDesc *p, float fault_tolerance);

API_PREFIX void QuantizedConvOpFree(QuantizedConvOpDesc *p);

API_PREFIX QuantizedFCOpDesc *QuantizedFCOpCreate(LAYOUT layout);

API_PREFIX void QuantizedFCOpSetupFCParameter(QuantizedFCOpDesc *p, size_t channel_out, size_t channel_in, float *src,
                                              float *bias, bool relu_fusion);

API_PREFIX void QuantizedFCOpQuantizeKernel(QuantizedFCOpDesc *p, float threshold);

API_PREFIX void QuantizedFCOpQuantizeData(QuantizedFCOpDesc *p, size_t batch_size, size_t channels, float *src,
                                          float sw_threshold);

API_PREFIX void QuantizedFCOpExecuteToDst(QuantizedFCOpDesc *p, float *dst, float fault_tolerance);

API_PREFIX void QuantizedFCOpSetupTargetBuffer(QuantizedFCOpDesc *p, float *dst);

API_PREFIX void QuantizedFCOpExecute(QuantizedFCOpDesc *p, float fault_tolerance);

API_PREFIX void QuantizedFCOpFree(QuantizedFCOpDesc *p);

// the following is TensorDescBased API.

API_PREFIX void QuantizedConvKernelDescInit(QuantizedTensor *quantized_tensor, size_t c_out, size_t c_in,
                                            size_t kernel_h, size_t kernel_w);

API_PREFIX void QuantizedConvKernelInit(QuantizedTensor *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                        size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

API_PREFIX void QuantizedConvKernelLoadFromModel(QuantizedTensor *quantized_tensor, int8_t *src, float *min,
                                                 float *max, size_t c_out, size_t c_in, size_t kernel_h,
                                                 size_t kernel_w, float threshold, LAYOUT layout);

API_PREFIX void QuantizedConvDataDescInit(QuantizedTensor *quantized_tensor, size_t c_in, size_t kernel_h,
                                          size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                          size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in,
                                          size_t w_in);

API_PREFIX void QuantizedConvDataInit(QuantizedTensor *quantized_tensor, float *src, size_t c_in, size_t kernel_h,
                                      size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                      size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in,
                                      float threshold, LAYOUT layout);

API_PREFIX void QuantizedConvKernelSumDescInit(FPTensor *fp_tensor, size_t c_out);

API_PREFIX void QuantizedConvKernelSumInit(FPTensor *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w);

API_PREFIX void QuantizedFCKernelDescInit(QuantizedTensor *quantized_tensor, size_t c_out, size_t c_in);

API_PREFIX void QuantizedFCKernelInit(QuantizedTensor *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                      float threshold, LAYOUT layout);

API_PREFIX void QuantizedFCKernelLoadFromModel(QuantizedTensor *quantized_tensor, int8_t *src, float *min,
                                               float *max, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

API_PREFIX void QuantizedFCDataDescInit(QuantizedTensor *quantized_tensor, size_t batch_size, size_t channel);

API_PREFIX void QuantizedFCDataInit(QuantizedTensor *quantized_tensor, float *src, size_t batch_size,
                                    size_t channel, float threshold, LAYOUT layout);

API_PREFIX void QuantizedFCKernelSumDescInit(FPTensor *fp_tensor, size_t c_out);

API_PREFIX void QuantizedFCKernelSumInit(FPTensor *fp_tensor, float *src, size_t c_out, size_t c_in);

API_PREFIX void MixPrecisionGEMM(LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k,
                                 float *ratio_a, float *ratio_b, float *kernel_sum, float *min_b, float *bias,
                                 size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
                                 float fault_tolerance, size_t pad_m, size_t pad_n);
#ifdef __cplusplus
}
#endif

#endif
