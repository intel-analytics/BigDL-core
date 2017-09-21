#ifndef BIGQUANT_H
#define BIGQUANT_H
#include <stddef.h>
#include <stdint.h>

typedef enum LAYOUT { NCHW = 0, NHWC = 1 } LAYOUT;
typedef enum CONV_ALGORITHM {
  AUTO_SELECT_CONV = 0,
  SHUFFLE_CONV = 1
} CONV_ALGORITHM;
typedef enum FC_ALGORITHM { AUTO_SELECT_FC = 0, SHUFFLE_FC = 1 } FC_ALGORITHM;

struct FPTensorDesc {
  void *data;
  size_t shape[4];
  size_t dim;
  size_t workspace_size;
};

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

struct QuantizedConvOp;
typedef struct QuantizedConvOp QuantizedConvOp;

struct QuantizedFCOp;
typedef struct QuantizedFCOp QuantizedFCOp;

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

#ifdef __cplusplus
extern "C" {
#endif

API_PREFIX int ManualRuntimeLoadLib(char *path);

QuantizedConvOp *QuantizedConvOpCreate();

API_PREFIX void QuantizedConvOpSetupConvParameter(
    QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
    size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
    size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
    size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo);

API_PREFIX void QuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight);

API_PREFIX void QuantizedConvOpExecute(QuantizedConvOp *p, float *dst,
                                       float *data, float *bias,
                                       size_t batch_size, size_t channel_in,
                                       size_t height_in, size_t width_in);

API_PREFIX void QuantizedConvOpFree(QuantizedConvOp *p);

QuantizedFCOp *QuantizedFCOpCreate();

API_PREFIX void QuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout,
                                              size_t channel_out,
                                              size_t channel_in,
                                              FC_ALGORITHM algo);

API_PREFIX void QuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight);

API_PREFIX void QuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data,
                                     float *bias, size_t batch_size,
                                     size_t channel_in);

API_PREFIX void QuantizedFCOpFree(QuantizedFCOp *p);

API_PREFIX void
QuantizedConvKernelDescInit(struct QuantizedTensorDesc *quantized_tensor,
                            size_t c_out, size_t c_in, size_t kernel_h,
                            size_t kernel_w);

API_PREFIX void
QuantizedConvKernelInit(struct QuantizedTensorDesc *quantized_tensor,
                        float *src, size_t c_out, size_t c_in, size_t kernel_h,
                        size_t kernel_w, float threshold, LAYOUT layout);

API_PREFIX void QuantizedConvKernelLoadFromModel(
    struct QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min,
    float *max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w,
    float threshold, LAYOUT layout);

API_PREFIX void
QuantizedConvDataDescInit(struct QuantizedTensorDesc *quantized_tensor,
                          size_t c_in, size_t kernel_h, size_t kernel_w,
                          size_t stride_h, size_t stride_w, size_t pad_h,
                          size_t pad_w, size_t dilation_h, size_t dilation_w,
                          size_t batch_size, size_t h_in, size_t w_in);

API_PREFIX void
QuantizedConvDataInit(struct QuantizedTensorDesc *quantized_tensor, float *src,
                      size_t c_in, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, size_t dilation_h, size_t dilation_w,
                      size_t batch_size, size_t h_in, size_t w_in,
                      float threshold, LAYOUT layout);

API_PREFIX void QuantizedConvKernelSumDescInit(struct FPTensorDesc *fp_tensor,
                                               size_t c_out);

API_PREFIX void QuantizedConvKernelSumInit(struct FPTensorDesc *fp_tensor,
                                           float *src, size_t n, size_t c,
                                           size_t h, size_t w);

API_PREFIX void MixPrecisionGEMM(
    LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n,
    size_t k, float *ratio_a, float *ratio_b, float *kernel_sum, float *min_b,
    float *bias, size_t batch_size, size_t channel_per_group, size_t height_out,
    size_t width_out, float fault_tolerance, size_t pad_m, size_t pad_n);

API_PREFIX void
QuantizedFCKernelDescInit(struct QuantizedTensorDesc *quantized_tensor,
                          size_t c_out, size_t c_in);

API_PREFIX void
QuantizedFCKernelInit(struct QuantizedTensorDesc *quantized_tensor, float *src,
                      size_t c_out, size_t c_in, float threshold,
                      LAYOUT layout);

API_PREFIX void QuantizedFCKernelLoadFromModel(
    struct QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min,
    float *max, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

API_PREFIX void
QuantizedFCDataDescInit(struct QuantizedTensorDesc *quantized_tensor,
                        size_t batch_size, size_t channel);

API_PREFIX void
QuantizedFCDataInit(struct QuantizedTensorDesc *quantized_tensor, float *src,
                    size_t batch_size, size_t channel, float threshold,
                    LAYOUT layout);

API_PREFIX void QuantizedFCKernelSumDescInit(struct FPTensorDesc *fp_tensor,
                                             size_t c_out);

API_PREFIX void QuantizedFCKernelSumInit(struct FPTensorDesc *fp_tensor,
                                         float *src, size_t c_out, size_t c_in);

API_PREFIX void FreeFPTensor(struct FPTensorDesc *p);

API_PREFIX void FreeQuantizedTensor(struct QuantizedTensorDesc *p);

#ifdef __cplusplus
}
#endif

#endif
