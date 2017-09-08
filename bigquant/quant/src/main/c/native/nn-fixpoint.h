#ifndef NN_FIXPOINT
#define NN_FIXPOINT
#include <stddef.h>
#include <stdint.h>

typedef enum LAYOUT {NCHW=0, NHWC=1} LAYOUT;

struct FPTensorDesc {
  void* data;
  size_t shape[4];
  size_t dim;
  size_t workspace_size;
};
typedef struct FPTensorDesc FPTensor;

struct FixTensorDesc {
  void* data;
  void* min;
  void* max;
  void* ratio;
  size_t shape[4];
  size_t ori_shape[4];
  size_t dim;
  size_t workspace_size_per_meta_info;;
  size_t workspace_size;
};
typedef struct FixTensorDesc FixTensor;


struct FixConvOp;
typedef struct FixConvOp FixConvOp;

struct FixConvOpDesc {
  LAYOUT layout;
  FixConvOp* op;
};
typedef struct FixConvOpDesc FixConvOpDesc;

struct FixFCOp;
typedef struct FixFCOp FixFCOp;

struct FixFCOpDesc {
  LAYOUT layout;
  FixFCOp* op;
};
typedef struct FixFCOpDesc FixFCOpDesc;

#ifdef WINDOWS
#define API_PREFIX __declspec(dllexport)
#else
#define API_PREFIX
#endif

extern "C"
{
  API_PREFIX void FreeMemory(void *p);

  API_PREFIX const char* GetGitCommit();

  API_PREFIX void Transpose(float *dst, float *src, size_t m, size_t n);

  API_PREFIX void LayoutTransform(LAYOUT dst_layout, LAYOUT src_layout, float *dst, float *src, size_t batch_size, size_t channel_size, size_t spatial_size);

  API_PREFIX FixConvOpDesc* FixConvOpCreate(LAYOUT layout);

  API_PREFIX void FixConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
          size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h, size_t pad_w, float *src, float *bias);

  API_PREFIX void FixFuseConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias,
          bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
          float *global_mean, float *global_variance, float epison, float *scale, float *shift);

  API_PREFIX void FixConvOpQuantizeKernel(FixConvOpDesc* p, float threshold);

  API_PREFIX void FixConvOpQuantizeData(FixConvOpDesc* p, size_t batch_size, size_t channels, size_t height_in, size_t width_in, float *src, float sw_threshold);

  API_PREFIX void FixConvOpExecuteToDst(FixConvOpDesc* p, float *dst, float fault_tolerance);

  API_PREFIX void FixConvOpSetupTargetBuffer(FixConvOpDesc* p, float* dst);

  API_PREFIX void FixConvOpExecute(FixConvOpDesc* p, float fault_tolerance);

  API_PREFIX void FixConvOpFree(FixConvOpDesc* p);

  API_PREFIX FixFCOpDesc* FixFCOpCreate(LAYOUT layout);

  API_PREFIX void FixFCOpSetupFCParameter(FixFCOpDesc* p, size_t channel_out, size_t channel_in, float *src, float *bias, bool relu_fusion);

  API_PREFIX void FixFCOpQuantizeKernel(FixFCOpDesc* p, float threshold);

  API_PREFIX void FixFCOpQuantizeData(FixFCOpDesc* p, size_t batch_size, size_t channels, float *src, float sw_threshold);

  API_PREFIX void FixFCOpExecuteToDst(FixFCOpDesc* p, float *dst, float fault_tolerance);

  API_PREFIX void FixFCOpSetupTargetBuffer(FixFCOpDesc* p, float* dst);

  API_PREFIX void FixFCOpExecute(FixFCOpDesc* p, float fault_tolerance);

  API_PREFIX void FixFCOpFree(FixFCOpDesc* p);

  // the following is Tensor Based API.

  API_PREFIX void FixConvKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w);

  API_PREFIX void FixConvKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

  API_PREFIX void FixConvKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

  API_PREFIX void FixConvDataDescInit(FixTensor *fix_tensor, size_t c_in, size_t kernel_h, size_t kernel_w, \
                          size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, \
                          size_t batch_size, size_t h_in, size_t w_in);

  API_PREFIX void FixConvDataInit(FixTensor *fix_tensor, float *src, size_t c_in, size_t kernel_h, size_t kernel_w, \
                          size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, \
                          size_t batch_size, size_t h_in, size_t w_in, float threshold, LAYOUT layout);

  API_PREFIX void FixConvKernelSumDescInit(FPTensor *fp_tensor, size_t c_out);

  API_PREFIX void FixConvKernelSumInit(FPTensor *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w);

  API_PREFIX void FixFCKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in);

  API_PREFIX void FixFCKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

  API_PREFIX void FixFCKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

  API_PREFIX void FixFCDataDescInit(FixTensor *fix_tensor, size_t batch_size, size_t channel);

  API_PREFIX void FixFCDataInit(FixTensor *fix_tensor, float *src, size_t batch_size, size_t channel, float threshold, LAYOUT layout);

  API_PREFIX void FixFCKernelSumDescInit(FPTensor *fp_tensor, size_t c_out);

  API_PREFIX void FixFCKernelSumInit(FPTensor *fp_tensor, float *src, size_t c_out, size_t c_in);

  API_PREFIX void InternalMixPrecisionGEMM(LAYOUT layout,\
    int8_t* pa, uint8_t* pb, float* pc, size_t m, size_t n, size_t k, \
    float* ratio_a, float* ratio_b, float* kernel_sum, float* min_b, float* bias, \
    size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out, \
    float fault_tolerance, size_t pad_m, size_t pad_n);
}
#endif


