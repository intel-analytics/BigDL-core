#include <mutex>
#include <dlfcn.h>
#include "nn-fixpoint.h"
#include "base.h"
#include "common.h"
#include "model.h"

std::mutex handler_mutex;

void *handler = NULL;

void (*FreeMemoryRT)(void *p);

const char* (*GetGitCommitRT)();

void (*TransposeRT)(float *dst, float *src, size_t m, size_t n);

void (*LayoutTransformRT)(LAYOUT dst_layout, LAYOUT src_layout, float *dst, float *src, size_t batch_size, size_t channel_size, size_t spatial_size);

FixConvOpDesc* (*FixConvOpCreateRT)(LAYOUT layout);

void (*FixConvOpSetupConvParameterRT)(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
          size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w, size_t pad_h, size_t pad_w, float *src, float *bias);

void (*FixFuseConvOpSetupConvParameterRT)(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias,
          bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
          float *global_mean, float *global_variance, float epison, float *scale, float *shift);

void (*FixConvOpQuantizeKernelRT)(FixConvOpDesc* p, float threshold);

void (*FixConvOpQuantizeDataRT)(FixConvOpDesc* p, size_t batch_size, size_t channels, size_t height_in, size_t width_in, float *src, float sw_threshold);

void (*FixConvOpExecuteToDstRT)(FixConvOpDesc* p, float *dst, float fault_tolerance);

void (*FixConvOpSetupTargetBufferRT)(FixConvOpDesc* p, float* dst);

void (*FixConvOpExecuteRT)(FixConvOpDesc* p, float fault_tolerance);

void (*FixConvOpFreeRT)(FixConvOpDesc* p);

void (*FixConvKernelDescInitRT)(FixTensor *fix_tensor, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w);

void (*FixConvKernelInitRT)(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

void (*FixConvKernelLoadFromModelRT)(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

void (*FixConvDataDescInitRT)(FixTensor *fix_tensor, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in);

void (*FixConvDataInitRT)(FixTensor *fix_tensor, float *src, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in, float threshold, LAYOUT layout);

void (*FixConvKernelSumDescInitRT)(FPTensor *fp_tensor, size_t c_out);

void (*FixConvKernelSumInitRT)(FPTensor *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w);

void (*InternalMixPrecisionGEMMRT)(LAYOUT layout, int8_t* pa, uint8_t* pb, float* pc, size_t m, size_t n, size_t k,
  float* ratio_a, float* ratio_b, float* kernel_sum, float* min_b, float* bias,
  size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
  float fault_tolerance, size_t pad_m, size_t pad_n);

FixFCOpDesc* (*FixFCOpCreateRT)(LAYOUT layout);

void (*FixFCOpSetupFCParameterRT)(FixFCOpDesc* p, size_t channel_out, size_t channel_in, float *src, float *bias, bool relu_fusion);

void (*FixFCOpQuantizeKernelRT)(FixFCOpDesc* p, float threshold);

void (*FixFCOpQuantizeDataRT)(FixFCOpDesc* p, size_t batch_size, size_t channels, float *src, float sw_threshold);

void (*FixFCOpExecuteToDstRT)(FixFCOpDesc* p, float *dst, float fault_tolerance);

void (*FixFCOpSetupTargetBufferRT)(FixFCOpDesc* p, float* dst);

void (*FixFCOpExecuteRT)(FixFCOpDesc* p, float fault_tolerance);

void (*FixFCOpFreeRT)(FixFCOpDesc* p);

void (*FixFCKernelDescInitRT)(FixTensor *fix_tensor, size_t c_out, size_t c_in);

void (*FixFCKernelInitRT)(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

void (*FixFCKernelLoadFromModelRT)(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, float threshold, LAYOUT layout);

void (*FixFCDataDescInitRT)(FixTensor *fix_tensor, size_t batch_size, size_t channel);

void (*FixFCDataInitRT)(FixTensor *fix_tensor, float *src, size_t batch_size, size_t channel, float threshold, LAYOUT layout);

void (*FixFCKernelSumDescInitRT)(FPTensor *fp_tensor, size_t c_out);

void (*FixFCKernelSumInitRT)(FPTensor *fp_tensor, float *src, size_t c_out, size_t c_in);

void __attribute__((constructor)) init_shared_library() {
  while (handler == NULL) {
    if (handler_mutex.try_lock()) {
      if (handler == NULL) {
        if (cpuid_support_feature(AVX_512)) {
          handler = dlopen("libnnfixpoint_avx512.so", RTLD_LAZY);
        } else if (cpuid_support_feature(AVX2_FMA)) {
          handler = dlopen("libnnfixpoint_avx2.so", RTLD_LAZY);
        } else if (cpuid_support_feature(SSE4_2)) {
          handler = dlopen("libnnfixpoint_sse42.so", RTLD_LAZY);
        } else {
          throw "unsupported machine. Target Machine should have SSE42+ ISA Support.\n";
        }
      }
      handler_mutex.unlock();
    }
  }
  FreeMemoryRT = dlsym(handler, "FreeMemory");
  GetGitCommitRT = dlsym(handler, "GetGitCommit");
  TransposeRT = dlsym(handler, "Transpose");
  LayoutTransformRT = dlsym(handler, "LayoutTransform");
  FixConvOpCreateRT = dlsym(handler, "FixConvOpCreate");
  FixConvOpSetupConvParameterRT = dlsym(handler, "FixConvOpSetupConvParameter");
  FixFuseConvOpSetupConvParameterRT = dlsym(handler, "FixFuseConvOpSetupConvParameter");
  FixConvOpQuantizeKernelRT = dlsym(handler, "FixConvOpQuantizeKernel");
  FixConvOpQuantizeDataRT = dlsym(handler, "FixConvOpQuantizeData");
  FixConvOpExecuteToDstRT = dlsym(handler, "FixConvOpExecuteToDst");
  FixConvOpSetupTargetBufferRT = dlsym(handler, "FixConvOpSetupTargetBuffer");
  FixConvOpExecuteRT = dlsym(handler, "FixConvOpExecute");
  FixConvOpFreeRT = dlsym(handler, "FixConvOpFree");
  FixConvKernelDescInitRT = dlsym(handler, "FixConvKernelDescInit");
  FixConvKernelInitRT = dlsym(handler, "FixConvKernelInit");
  FixConvKernelLoadFromModelRT = dlsym(handler, "FixConvKernelLoadFromModel");
  FixConvDataDescInitRT = dlsym(handler, "FixConvDataDescInit");
  FixConvDataInitRT = dlsym(handler, "FixConvDataInit");
  FixConvKernelSumDescInitRT = dlsym(handler, "FixConvKernelSumDescInit");
  FixConvKernelSumInitRT = dlsym(handler, "FixConvKernelSumInit");
  InternalMixPrecisionGEMMRT = dlsym(handler, "InternalMixPrecisionGEMM");
  FixFCOpCreateRT = dlsym(handler, "FixFCOpCreate");
  FixFCOpSetupFCParameterRT = dlsym(handler, "FixFCOpSetupFCParameter");
  FixFCOpQuantizeKernelRT = dlsym(handler, "FixFCOpQuantizeKernel");
  FixFCOpQuantizeDataRT = dlsym(handler, "FixFCOpQuantizeData");
  FixFCOpExecuteToDstRT = dlsym(handler, "FixFCOpExecuteToDst");
  FixFCOpSetupTargetBufferRT = dlsym(handler, "FixFCOpSetupTargetBuffer");
  FixFCOpExecuteRT = dlsym(handler, "FixFCOpExecute");
  FixFCOpFreeRT = dlsym(handler, "FixFCOpFree");
  FixFCKernelDescInitRT = dlsym(handler, "FixFCKernelDescInit");
  FixFCKernelInitRT = dlsym(handler, "FixFCKernelInit");
  FixFCKernelLoadFromModelRT = dlsym(handler, "FixFCKernelLoadFromModel");
  FixFCDataDescInitRT = dlsym(handler, "FixFCDataDescInit");
  FixFCDataInitRT = dlsym(handler, "FixFCDataInit");
  FixFCKernelSumDescInitRT = dlsym(handler, "FixFCKernelSumDescInit");
  FixFCKernelSumInitRT = dlsym(handler, "FixFCKernelSumInit");
}

void __attribute__((destructor)) free_shared_library() {
  while (handler != NULL) {
    if (handler_mutex.try_lock()) {
      if (handler != NULL) {
        dlclose(handler);
        handler = NULL;
      }
      handler_mutex.unlock();
    }
  }
}


const char* GetGitCommit() {
  return GetGitCommitRT();
}

void Transpose(float *dst, float *src, size_t m, size_t n) {
  TransposeRT(dst, src, m, n);
}

void LayoutTransform(LAYOUT dst_layout, LAYOUT src_layout, float *dst, float *src, size_t batch_size, size_t channel_size, size_t spatial_size) {
  LayoutTransformRT(dst_layout, src_layout, dst, src, batch_size, channel_size, spatial_size);
}

void FreeMemory(void *p) {
  FreeMemoryRT(p);
}

FixConvOpDesc* FixConvOpCreate(LAYOUT layout) {
  return FixConvOpCreateRT(layout);
}

void FixFuseConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w,
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias,
          bool conv_relu_fusion, bool conv_bn_fusion, bool conv_bn_relu_fusion, bool conv_relu_bn_fusion,
          float *global_mean, float *global_variance, float epison, float *scale, float *shift) {
  FixFuseConvOpSetupConvParameterRT(p, channel_out, channel_in, group, kernel_h, kernel_w,
                                  stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias,
                                  conv_relu_fusion, conv_bn_fusion, conv_bn_relu_fusion, conv_relu_bn_fusion,
                                  global_mean, global_variance, epison, scale, shift);
}

void FixConvOpSetupConvParameter(FixConvOpDesc* p, size_t channel_out, size_t channel_in, size_t group, size_t kernel_h, size_t kernel_w, \
          size_t stride_h, size_t stride_w, size_t dialation_h, size_t dialation_w, size_t pad_h, size_t pad_w, float *src, float *bias) {
  FixConvOpSetupConvParameterRT(p, channel_out, channel_in, group, kernel_h, kernel_w,
          stride_h, stride_w, dialation_h, dialation_w, pad_h, pad_w, src, bias);
}

void FixConvOpQuantizeKernel(FixConvOpDesc* p, float threshold) {
  FixConvOpQuantizeKernelRT(p, threshold);
}

void FixConvOpQuantizeData(FixConvOpDesc* p, size_t batch_size, size_t channels, size_t height_in, size_t width_in, float *src, float sw_threshold) {
  FixConvOpQuantizeDataRT(p, batch_size, channels, height_in, width_in, src, sw_threshold);
}

void FixConvOpExecuteToDst(FixConvOpDesc* p, float* dst, float fault_tolerance) {
  FixConvOpExecuteToDstRT(p, dst, fault_tolerance);
}

void FixConvOpSetupTargetBuffer(FixConvOpDesc* p, float* dst) {
  FixConvOpSetupTargetBufferRT(p, dst);
}

void FixConvOpExecute(FixConvOpDesc* p, float fault_tolerance) {
  FixConvOpExecuteRT(p, fault_tolerance);
}


void FixConvOpFree(FixConvOpDesc* p) {
  FixConvOpFreeRT(p);
}

void FixConvKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w) {
  FixConvKernelDescInitRT(fix_tensor, c_out, c_in, kernel_h, kernel_w);
}

void FixConvKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  FixConvKernelInitRT(fix_tensor, src, c_out, c_in, kernel_h, kernel_w, threshold, layout);
}

void FixConvKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  FixConvKernelLoadFromModelRT(fix_tensor, src, min, max, c_out, c_in, kernel_h, kernel_w, threshold, layout);
}

void FixConvDataDescInit(FixTensor *fix_tensor, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in) {
  FixConvDataDescInitRT(fix_tensor, c_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, batch_size, h_in, w_in);
}

void FixConvDataInit(FixTensor *fix_tensor, float *src, size_t c_in, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in, float threshold, LAYOUT layout) {
  FixConvDataInitRT(fix_tensor, src, c_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, batch_size, h_in, w_in, threshold, layout);
}

void FixConvKernelSumDescInit(FPTensor *fp_tensor, size_t c_out) {
  FixConvKernelSumDescInitRT(fp_tensor, c_out);
}

void FixConvKernelSumInit(FPTensor *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w) {
  FixConvKernelSumInitRT(fp_tensor, src, n, c, h, w);
}

void InternalMixPrecisionGEMM(LAYOUT layout, int8_t* pa, uint8_t* pb, float* pc, size_t m, size_t n, size_t k,
  float* ratio_a, float* ratio_b, float* kernel_sum, float* min_b, float* bias,
  size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
  float fault_tolerance, size_t pad_m, size_t pad_n) {
  InternalMixPrecisionGEMMRT(layout, pa, pb, pc, m, n, k,
  ratio_a, ratio_b, kernel_sum, min_b, bias,
  batch_size, channel_per_group, height_out, width_out,
  fault_tolerance, pad_m, pad_n);
}

FixFCOpDesc* FixFCOpCreate(LAYOUT layout) {
  return FixFCOpCreateRT(layout);
}

void FixFCOpSetupFCParameter(FixFCOpDesc* p, size_t channel_out, size_t channel_in, float *src, float *bias, bool relu_fusion) {
  FixFCOpSetupFCParameterRT(p, channel_out, channel_in, src, bias, relu_fusion);
}

void FixFCOpQuantizeKernel(FixFCOpDesc* p, float threshold) {
  FixFCOpQuantizeKernelRT(p, threshold);
}

void FixFCOpQuantizeData(FixFCOpDesc* p, size_t batch_size, size_t channels, float *src, float sw_threshold) {
  FixFCOpQuantizeDataRT(p, batch_size, channels, src, sw_threshold);
}

void FixFCOpExecuteToDst(FixFCOpDesc* p, float* dst, float fault_tolerance) {
  FixFCOpExecuteToDstRT(p, dst, fault_tolerance);
}

void FixFCOpSetupTargetBuffer(FixFCOpDesc* p, float* dst) {
  FixFCOpSetupTargetBufferRT(p, dst);
}

void FixFCOpExecute(FixFCOpDesc* p, float fault_tolerance) {
  FixFCOpExecuteRT(p, fault_tolerance);
}

void FixFCOpFree(FixFCOpDesc* p) {
  FixFCOpFreeRT(p);
}

void FixFCKernelDescInit(FixTensor *fix_tensor, size_t c_out, size_t c_in) {
  FixFCKernelDescInitRT(fix_tensor, c_out, c_in);
}

void FixFCKernelInit(FixTensor *fix_tensor, float *src, size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  FixFCKernelInitRT(fix_tensor, src, c_out, c_in, threshold, layout);
}

void FixFCKernelLoadFromModel(FixTensor *fix_tensor, int8_t *src, float *min, float* max, size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  FixFCKernelLoadFromModelRT(fix_tensor, src, min, max, c_out, c_in, threshold, layout);
}

void FixFCDataDescInit(FixTensor *fix_tensor, size_t batch_size, size_t channel) {
  FixFCDataDescInitRT(fix_tensor, batch_size, channel);
}

void FixFCDataInit(FixTensor *fix_tensor, float *src, size_t batch_size, size_t channel, float threshold, LAYOUT layout) {
  FixFCDataInitRT(fix_tensor, src, batch_size, channel, threshold, layout);
}

void FixFCKernelSumDescInit(FPTensor *fp_tensor, size_t c_out) {
  FixFCKernelSumDescInitRT(fp_tensor, c_out);
}

void FixFCKernelSumInit(FPTensor *fp_tensor, float *src, size_t c_out, size_t c_in) {
  FixFCKernelSumInitRT(fp_tensor, src, c_out, c_in);
}
