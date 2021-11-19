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

#if defined(WINDOWS)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include "bigquant.h"
#include "base.h"
#include "common.h"

#if defined(WINDOWS)
HINSTANCE handler = NULL;
#else
void *handler = NULL;
#endif

QuantizedConvOp *(*QuantizedConvOpCreateRT)();

void (*QuantizedConvOpSetupConvParameterRT)(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                            size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h,
                                            size_t stride_w, size_t pad_h, size_t pad_w, size_t dialation_h,
                                            size_t dialation_w, size_t fusion_mask, CONV_ALGORITHM algo);

void (*QuantizedConvOpInitWeightRT)(QuantizedConvOp *p, float *weight);

void (*QuantizedConvOpExecuteRT)(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                                 size_t channel_in, size_t height_in, size_t width_in);

void (*QuantizedConvOpFreeRT)(QuantizedConvOp *p);

QuantizedFCOp *(*QuantizedFCOpCreateRT)();

void (*QuantizedFCOpSetupFCParameterRT)(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                        FC_ALGORITHM algo);

void (*QuantizedFCOpInitWeightRT)(QuantizedFCOp *p, float *weight);

void (*QuantizedFCOpExecuteRT)(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                               size_t channel_in);

void (*QuantizedFCOpFreeRT)(QuantizedFCOp *p);

void (*QuantizedConvKernelDescInitRT)(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in, size_t kernel_h,
                                      size_t kernel_w);

void (*QuantizedConvKernelInitRT)(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                  size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout);

void (*QuantizedConvKernelLoadFromModelRT)(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                           size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold,
                                           LAYOUT layout);

void (*QuantizedConvDataDescInitRT)(QuantizedTensorDesc *quantized_tensor, size_t c_in, size_t kernel_h,
                                    size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                    size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in);

void (*QuantizedConvDataInitRT)(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_in, size_t kernel_h,
                                size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in,
                                float threshold, LAYOUT layout);

void (*QuantizedConvKernelSumDescInitRT)(FPTensorDesc *fp_tensor, size_t c_out);

void (*QuantizedConvKernelSumInitRT)(FPTensorDesc *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w);

void (*MixPrecisionGEMMRT)(LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k,
                           float *ratio_a, float *ratio_b, float *kernel_sum, float *min_b, float *bias,
                           size_t batch_size, size_t channel_per_group, size_t height_out, size_t width_out,
                           float fault_tolerance, size_t pad_m, size_t pad_n);

void (*QuantizedFCKernelDescInitRT)(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in);

void (*QuantizedFCKernelInitRT)(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                                float threshold, LAYOUT layout);

void (*QuantizedFCKernelLoadFromModelRT)(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                         size_t c_out, size_t c_in, float threshold, LAYOUT layout);

void (*QuantizedFCDataDescInitRT)(QuantizedTensorDesc *quantized_tensor, size_t batch_size, size_t channel);

void (*QuantizedFCDataInitRT)(QuantizedTensorDesc *quantized_tensor, float *src, size_t batch_size, size_t channel,
                              float threshold, LAYOUT layout);

void (*QuantizedFCKernelSumDescInitRT)(FPTensorDesc *fp_tensor, size_t c_out);

void (*QuantizedFCKernelSumInitRT)(FPTensorDesc *fp_tensor, float *src, size_t c_out, size_t c_in);

void (*FreeFPTensorRT)(struct FPTensorDesc *p);

void (*FreeQuantizedTensorRT)(struct QuantizedTensorDesc *p);

void BindSymbol() {
#if defined(WINDOWS)
#define BINDSYMBOL GetProcAddress
#else
#define BINDSYMBOL dlsym
#endif
  QuantizedConvOpCreateRT =
      reinterpret_cast<QuantizedConvOp *(*)()>(BINDSYMBOL(handler, "InternalQuantizedConvOpCreate"));
  QuantizedConvOpSetupConvParameterRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, LAYOUT, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                                size_t, size_t, size_t, size_t, size_t, CONV_ALGORITHM)>(
          BINDSYMBOL(handler, "InternalQuantizedConvOpSetupConvParameter"));
  QuantizedConvOpInitWeightRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, float *)>(BINDSYMBOL(handler, "InternalQuantizedConvOpInitWeight"));
  QuantizedConvOpExecuteRT =
      reinterpret_cast<void (*)(QuantizedConvOp *, float *, float *, float *, size_t, size_t, size_t, size_t)>(
          BINDSYMBOL(handler, "InternalQuantizedConvOpExecute"));
  QuantizedConvOpFreeRT =
      reinterpret_cast<void (*)(QuantizedConvOp *)>(BINDSYMBOL(handler, "InternalQuantizedConvOpFree"));
  QuantizedFCOpCreateRT = reinterpret_cast<QuantizedFCOp *(*)()>(BINDSYMBOL(handler, "InternalQuantizedFCOpCreate"));
  QuantizedFCOpSetupFCParameterRT = reinterpret_cast<void (*)(QuantizedFCOp *, LAYOUT, size_t, size_t, FC_ALGORITHM)>(
      BINDSYMBOL(handler, "InternalQuantizedFCOpSetupFCParameter"));
  QuantizedFCOpInitWeightRT =
      reinterpret_cast<void (*)(QuantizedFCOp *, float *)>(BINDSYMBOL(handler, "InternalQuantizedFCOpInitWeight"));
  QuantizedFCOpExecuteRT = reinterpret_cast<void (*)(QuantizedFCOp *, float *, float *, float *, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedFCOpExecute"));
  QuantizedFCOpFreeRT = reinterpret_cast<void (*)(QuantizedFCOp *)>(BINDSYMBOL(handler, "InternalQuantizedFCOpFree"));
  QuantizedConvKernelDescInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, size_t, size_t, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedConvKernelDescInit"));
  QuantizedConvKernelInitRT =
      reinterpret_cast<void (*)(QuantizedTensorDesc *, float *, size_t, size_t, size_t, size_t, float, LAYOUT)>(
          BINDSYMBOL(handler, "InternalQuantizedConvKernelInit"));
  QuantizedConvKernelLoadFromModelRT =
      reinterpret_cast<void (*)(QuantizedTensorDesc *, int8_t *, float *, float *, size_t, size_t, size_t, size_t,
                                float, LAYOUT)>(BINDSYMBOL(handler, "InternalQuantizedConvKernelLoadFromModel"));
  QuantizedConvDataDescInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, size_t, size_t, size_t, size_t, size_t,
                                                          size_t, size_t, size_t, size_t, size_t, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedConvDataDescInit"));
  QuantizedConvDataInitRT =
      reinterpret_cast<void (*)(QuantizedTensorDesc *, float *, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                                size_t, size_t, size_t, size_t, size_t, float, LAYOUT)>(
          BINDSYMBOL(handler, "InternalQuantizedConvDataInit"));
  QuantizedConvKernelSumDescInitRT =
      reinterpret_cast<void (*)(FPTensorDesc *, size_t)>(BINDSYMBOL(handler, "InternalQuantizedConvKernelSumDescInit"));
  QuantizedConvKernelSumInitRT = reinterpret_cast<void (*)(FPTensorDesc *, float *, size_t, size_t, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedConvKernelSumInit"));
  MixPrecisionGEMMRT =
      reinterpret_cast<void (*)(LAYOUT, int8_t *, uint8_t *, float *, size_t, size_t, size_t, float *, float *, float *,
                                float *, float *, size_t, size_t, size_t, size_t, float, size_t, size_t)>(
          BINDSYMBOL(handler, "InternalMixPrecisionGEMM"));
  QuantizedFCKernelDescInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedFCKernelDescInit"));
  QuantizedFCKernelInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, float *, size_t, size_t, float, LAYOUT)>(
      BINDSYMBOL(handler, "InternalQuantizedFCKernelInit"));
  QuantizedFCKernelLoadFromModelRT =
      reinterpret_cast<void (*)(QuantizedTensorDesc *, int8_t *, float *, float *, size_t, size_t, float, LAYOUT)>(
          BINDSYMBOL(handler, "InternalQuantizedFCKernelLoadFromModel"));
  QuantizedFCDataDescInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedFCDataDescInit"));
  QuantizedFCDataInitRT = reinterpret_cast<void (*)(QuantizedTensorDesc *, float *, size_t, size_t, float, LAYOUT)>(
      BINDSYMBOL(handler, "InternalQuantizedFCDataInit"));
  QuantizedFCKernelSumDescInitRT =
      reinterpret_cast<void (*)(FPTensorDesc *, size_t)>(BINDSYMBOL(handler, "InternalQuantizedFCKernelSumDescInit"));
  QuantizedFCKernelSumInitRT = reinterpret_cast<void (*)(FPTensorDesc *, float *, size_t, size_t)>(
      BINDSYMBOL(handler, "InternalQuantizedFCKernelSumInit"));
  FreeFPTensorRT = reinterpret_cast<void (*)(FPTensorDesc *)>(BINDSYMBOL(handler, "InternalFreeFPTensor"));
  FreeQuantizedTensorRT =
      reinterpret_cast<void (*)(QuantizedTensorDesc *)>(BINDSYMBOL(handler, "InternalFreeQuantizedTensor"));
#undef BINDSYMBOL
}

int ManualRuntimeLoadLib(char *path) {
#if defined(MANUAL_LOAD)
  char lib_path[300];
  strncpy(lib_path, path, 200);
#if defined(WINDOWS)
  char *ext = ".dll";
#elif defined(__APPLE__)
  char *ext = ".dylib";
#else
  const char *ext = ".so";
#endif
  if (handler == NULL) {
    if (cpuid_support_feature(AVX2_FMA)) {
      strncat(lib_path, "/libbigquant_avx2", 100);
    } else if (cpuid_support_feature(SSE4_2)) {
      strncat(lib_path, "/libbigquant_sse42", 100);
    }
    #if ! defined(WINDOWS) && ! defined(__APPLE__)
    if (cpuid_support_feature(AVX_512)) {
      strncat(lib_path, "/libbigquant_avx512", 100);
    }
    #endif
    if (handler == NULL) {
      fprintf(stderr, "Unsupported ISA. Bigquant supports Instruction Set from SSE42 to AVX512.\n");
      return -1;
    }
    strncat(lib_path, ext, 100);
#if defined(WINDOWS)
    handler = LoadLibrary(lib_path);
#else   // WINSOWS
    handler = dlopen(lib_path, RTLD_NOW | RTLD_NODELETE);
#endif  // WINDOWS
    if (handler == NULL) {
      fprintf(stderr, "%s failed to be loaded.\n", lib_path);
      return -2;
    }
  }
  BindSymbol();
#ifndef WINDOWS
  if (handler != NULL) {
    dlclose(handler);
    handler = NULL;
  }
#endif  // WINDOWS
  return 0;
#else   // MANUAL_LOAD
  std::cerr << "Useless Function. Please build with -DMANUAL_LOAD to enable this function." << std::endl;
  return -3;
#endif  // MANUAL_LOAD
}

void __attribute__((constructor)) init_shared_library() {
#ifndef MANUAL_LOAD
  std::string lib_path;
#if defined(WINDOWS)
  std::string ext = ".dll";
#elif defined(__APPLE__)
  std::string ext = ".dylib";
#else
  std::string ext = ".so";
#endif
  if (handler == NULL) {
    if (cpuid_support_feature(AVX_512)) {
      lib_path = "libbigquant_avx512";
    } else if (cpuid_support_feature(AVX2_FMA)) {
      lib_path = "libbigquant_avx2";
    } else if (cpuid_support_feature(SSE4_2)) {
      lib_path = "libbigquant_sse42";
    }
    #if ! defined(WINDOWS) && ! defined(__APPLE__)
    if (cpuid_support_feature(AVX_512)) {
      strncat(lib_path, "/libbigquant_avx512", 100);
    }
    #endif
    if (handler == NULL) {
      std::cerr << "Unsupported ISA. Bigquant supports Instruction Set from SSE42 to AVX512.\n" << std::endl;
      exit(-1);
    }
    lib_path += ext;
#if defined(WINDOWS)
    handler = LoadLibrary(lib_path.c_str());
#else   // WINSOWS
    handler = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_NODELETE);
#endif  // WINDOWS
    if (handler == NULL) {
      std::cerr << lib_path.c_str() << " failed to be loaded." << std::endl;
      exit(-1);
    }
  }
  BindSymbol();
#ifndef WINDOWS
  if (handler != NULL) {
    dlclose(handler);
    handler = NULL;
  }
#endif  // WINDOWS
#endif
}

void __attribute__((destructor)) free_shared_library() {
#ifndef MANUAL_LOAD
  if (handler != NULL) {
#if defined(WINDOWS)
    FreeLibrary(handler);
#else   // WINDOWS
    dlclose(handler);
#endif  // WINDOWS
  }
#endif
}

QuantizedConvOp *QuantizedConvOpCreate() {
  return QuantizedConvOpCreateRT();
}

void QuantizedConvOpSetupConvParameter(QuantizedConvOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                       size_t group, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                                       size_t pad_h, size_t pad_w, size_t dialation_h, size_t dialation_w,
                                       size_t fusion_mask, CONV_ALGORITHM algo) {
  QuantizedConvOpSetupConvParameterRT(p, layout, channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w,
                                      pad_h, pad_w, dialation_h, dialation_w, fusion_mask, algo);
}

void QuantizedConvOpInitWeight(QuantizedConvOp *p, float *weight) {
  QuantizedConvOpInitWeightRT(p, weight);
}

void QuantizedConvOpExecute(QuantizedConvOp *p, float *dst, float *data, float *bias, size_t batch_size,
                            size_t channel_in, size_t height_in, size_t width_in) {
  QuantizedConvOpExecuteRT(p, dst, data, bias, batch_size, channel_in, height_in, width_in);
}

void QuantizedConvOpFree(QuantizedConvOp *p) {
  QuantizedConvOpFreeRT(p);
}

QuantizedFCOp *QuantizedFCOpCreate() {
  return QuantizedFCOpCreateRT();
}

void QuantizedFCOpSetupFCParameter(QuantizedFCOp *p, LAYOUT layout, size_t channel_out, size_t channel_in,
                                   FC_ALGORITHM algo) {
  QuantizedFCOpSetupFCParameterRT(p, layout, channel_out, channel_in, algo);
}

void QuantizedFCOpInitWeight(QuantizedFCOp *p, float *weight) {
  QuantizedFCOpInitWeightRT(p, weight);
}

void QuantizedFCOpExecute(QuantizedFCOp *p, float *dst, float *data, float *bias, size_t batch_size,
                          size_t channel_in) {
  QuantizedFCOpExecuteRT(p, dst, data, bias, batch_size, channel_in);
}

void QuantizedFCOpFree(QuantizedFCOp *p) {
  QuantizedFCOpFreeRT(p);
}

void QuantizedConvKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in, size_t kernel_h,
                                 size_t kernel_w) {
  QuantizedConvKernelDescInitRT(quantized_tensor, c_out, c_in, kernel_h, kernel_w);
}

void QuantizedConvKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                             size_t kernel_h, size_t kernel_w, float threshold, LAYOUT layout) {
  QuantizedConvKernelInitRT(quantized_tensor, src, c_out, c_in, kernel_h, kernel_w, threshold, layout);
}

void QuantizedConvKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                      size_t c_out, size_t c_in, size_t kernel_h, size_t kernel_w, float threshold,
                                      LAYOUT layout) {
  QuantizedConvKernelLoadFromModelRT(quantized_tensor, src, min, max, c_out, c_in, kernel_h, kernel_w, threshold,
                                     layout);
}

void QuantizedConvDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_in, size_t kernel_h, size_t kernel_w,
                               size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t dilation_h,
                               size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in) {
  QuantizedConvDataDescInitRT(quantized_tensor, c_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                              dilation_w, batch_size, h_in, w_in);
}

void QuantizedConvDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_in, size_t kernel_h,
                           size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                           size_t dilation_h, size_t dilation_w, size_t batch_size, size_t h_in, size_t w_in,
                           float threshold, LAYOUT layout) {
  QuantizedConvDataInitRT(quantized_tensor, src, c_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                          dilation_w, batch_size, h_in, w_in, threshold, layout);
}

void QuantizedConvKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out) {
  QuantizedConvKernelSumDescInitRT(fp_tensor, c_out);
}

void QuantizedConvKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t n, size_t c, size_t h, size_t w) {
  QuantizedConvKernelSumInitRT(fp_tensor, src, n, c, h, w);
}

void MixPrecisionGEMM(LAYOUT layout, int8_t *pa, uint8_t *pb, float *pc, size_t m, size_t n, size_t k, float *ratio_a,
                      float *ratio_b, float *kernel_sum, float *min_b, float *bias, size_t batch_size,
                      size_t channel_per_group, size_t height_out, size_t width_out, float fault_tolerance,
                      size_t pad_m, size_t pad_n) {
  MixPrecisionGEMMRT(layout, pa, pb, pc, m, n, k, ratio_a, ratio_b, kernel_sum, min_b, bias, batch_size,
                     channel_per_group, height_out, width_out, fault_tolerance, pad_m, pad_n);
}

void QuantizedFCKernelDescInit(QuantizedTensorDesc *quantized_tensor, size_t c_out, size_t c_in) {
  QuantizedFCKernelDescInitRT(quantized_tensor, c_out, c_in);
}

void QuantizedFCKernelInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t c_out, size_t c_in,
                           float threshold, LAYOUT layout) {
  QuantizedFCKernelInitRT(quantized_tensor, src, c_out, c_in, threshold, layout);
}

void QuantizedFCKernelLoadFromModel(QuantizedTensorDesc *quantized_tensor, int8_t *src, float *min, float *max,
                                    size_t c_out, size_t c_in, float threshold, LAYOUT layout) {
  QuantizedFCKernelLoadFromModelRT(quantized_tensor, src, min, max, c_out, c_in, threshold, layout);
}

void QuantizedFCDataDescInit(QuantizedTensorDesc *quantized_tensor, size_t batch_size, size_t channel) {
  QuantizedFCDataDescInitRT(quantized_tensor, batch_size, channel);
}

void QuantizedFCDataInit(QuantizedTensorDesc *quantized_tensor, float *src, size_t batch_size, size_t channel,
                         float threshold, LAYOUT layout) {
  QuantizedFCDataInitRT(quantized_tensor, src, batch_size, channel, threshold, layout);
}

void QuantizedFCKernelSumDescInit(FPTensorDesc *fp_tensor, size_t c_out) {
  QuantizedFCKernelSumDescInitRT(fp_tensor, c_out);
}

void QuantizedFCKernelSumInit(FPTensorDesc *fp_tensor, float *src, size_t c_out, size_t c_in) {
  QuantizedFCKernelSumInitRT(fp_tensor, src, c_out, c_in);
}

void FreeFPTensor(struct FPTensorDesc *p) {
  FreeFPTensorRT(p);
}

void FreeQuantizedTensor(struct QuantizedTensorDesc *p) {
  FreeQuantizedTensorRT(p);
}
