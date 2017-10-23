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

#include <stdio.h>
#include <stdlib.h>

#include "bigquant.h"
#include "com_intel_analytics_bigdl_bigquant_BigQuant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QuantizedTensorDesc QuantizedTensor;
typedef struct FPTensorDesc FPTensor;

JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_printHello(JNIEnv *env,
                                                            jclass cls)
{
  printf("Hello BigQuant!");
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelDescInit
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelDescInit(
    JNIEnv *env, jclass cls, jint c_out, jint c_in, jint kernel_h,
    jint kernel_w)
{
  QuantizedTensor *tmp = (QuantizedTensor *)malloc(sizeof(QuantizedTensor));
  QuantizedConvKernelDescInit(tmp, c_out, c_in, kernel_h, kernel_w);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelInit
 * Signature: (J[FIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelInit(
    JNIEnv *env, jclass cls, jlong tensor, jfloatArray src, jint srcOffset,
    jint c_out, jint c_in, jint kernel_h, jint kernel_w, jfloat threshold,
    jint layout)
{
  QuantizedTensor *j_tensor = (QuantizedTensor *)tensor;

  jfloat *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  QuantizedConvKernelInit(j_tensor, jni_src + srcOffset, c_out, c_in, kernel_h,
                          kernel_w, threshold, layout);
  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelLoadFromModel
 * Signature: (J[BI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelLoadFromModel(
    JNIEnv *env, jclass cls, jlong tensor, jbyteArray src, jint srcOffset,
    jfloatArray min, jfloatArray max, jint c_out, jint c_in, jint kernel_h,
    jint kernel_w, jfloat threshold, jint layout)
{
  QuantizedTensor *j_tensor = (QuantizedTensor *)tensor;
  jbyte *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  jfloat *jni_min = (*env)->GetPrimitiveArrayCritical(env, min, JNI_FALSE);
  jfloat *jni_max = (*env)->GetPrimitiveArrayCritical(env, max, JNI_FALSE);

  QuantizedConvKernelLoadFromModel(j_tensor, jni_src, jni_min, jni_max, c_out,
                                   c_in, kernel_h, kernel_w, threshold, layout);

  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, min, jni_min, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, max, jni_max, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvDataDescInit
 * Signature: (IIIIIIIIIIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvDataDescInit(
    JNIEnv *env, jclass cls, jint c_in, jint kernel_h, jint kernel_w,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint dilation_h,
    jint dilation_w, jint batch_size, jint h_in, jint w_in)
{
  QuantizedTensor *tmp = (QuantizedTensor *)malloc(sizeof(QuantizedTensor));
  QuantizedConvDataDescInit(tmp, c_in, kernel_h, kernel_w, stride_h, stride_w,
                            pad_h, pad_w, dilation_h, dilation_w, batch_size,
                            h_in, w_in);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvDataInit
 * Signature: (J[FIIIIIIIIIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvDataInit(
    JNIEnv *env, jclass cls, jlong tensor, jfloatArray src, jint srcOffset,
    jint c_in, jint kernel_h, jint kernel_w, jint stride_h, jint stride_w,
    jint pad_h, jint pad_w, jint dilation_h, jint dilation_w, jint batch_size,
    jint h_in, jint w_in, jfloat threshold, jint layout)
{
  QuantizedTensor *j_tensor = (QuantizedTensor *)tensor;

  jfloat *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  QuantizedConvDataInit(j_tensor, jni_src + srcOffset, c_in, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dilation_h,
                        dilation_w, batch_size, h_in, w_in, threshold, layout);
  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelSumDescInit
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelSumDescInit(
    JNIEnv *env, jclass cls, jint c_out)
{
  FPTensor *tmp = (FPTensor *)malloc(sizeof(FPTensor));
  QuantizedConvKernelSumDescInit(tmp, c_out);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelSumInit(
    JNIEnv *env, jclass cls, jlong fp_tensor, jfloatArray src, jint srcOffset,
    jint n, jint c, jint h, jint w)
{
  FPTensor *j_fp_tensor = (FPTensor *)fp_tensor;

  jfloat *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  QuantizedConvKernelSumInit(j_fp_tensor, jni_src + srcOffset, n, c, h, w);
  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    MixPrecisionGEMM
 * Signature: (IJIJ[FIIII[FI[FIIIIIIF)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_MixPrecisionGEMM(
    JNIEnv *env, jclass cls, jint layout, jlong pa, jlong pb, jfloatArray pc,
    jint pcOffset, jfloatArray kernel_sum, jint kernel_sum_offset,
    jfloatArray bias, jint biasOffset, jint batch_size, jint channel_per_group,
    jint height_out, jint width_out, jfloat fault_tolerance)
{
  QuantizedTensor *jni_pa = (QuantizedTensor *)pa;
  QuantizedTensor *jni_pb = (QuantizedTensor *)pb;

  jfloat *jni_pc = (*env)->GetPrimitiveArrayCritical(env, pc, JNI_FALSE);
  jfloat *jni_bias = (*env)->GetPrimitiveArrayCritical(env, bias, JNI_FALSE);
  jfloat *jni_kernel_sum =
      (*env)->GetPrimitiveArrayCritical(env, kernel_sum, JNI_FALSE);

  MixPrecisionGEMM(
      layout, jni_pa->data, jni_pb->data, jni_pc + pcOffset, jni_pa->shape[0],
      jni_pb->shape[0], jni_pb->shape[1], jni_pa->ratio, jni_pb->ratio,
      jni_kernel_sum + kernel_sum_offset, jni_pb->min, jni_bias + biasOffset,
      batch_size, channel_per_group, height_out, width_out, fault_tolerance,
      jni_pa->shape[0] - jni_pa->ori_shape[0],
      jni_pb->shape[0] - jni_pb->ori_shape[0]);

  (*env)->ReleasePrimitiveArrayCritical(env, pc, jni_pc, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, bias, jni_bias, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, kernel_sum, jni_kernel_sum, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FreeMemory
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FreeMemory(JNIEnv *env,
                                                            jclass cls,
                                                            jlong ptr)
{
  QuantizedTensor *jni_ptr = (QuantizedTensor *)ptr;
  FreeQuantizedTensor(jni_ptr);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCKernelDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCKernelDescInit(JNIEnv *env,
                                                                  jclass cls,
                                                                  jint c_out,
                                                                  jint c_in)
{
  QuantizedTensor *tmp = (QuantizedTensor *)malloc(sizeof(QuantizedTensor));
  QuantizedFCKernelDescInit(tmp, c_out, c_in);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCKernelLoadFromModel
 * Signature: (J[B[F[FIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCKernelLoadFromModel(
    JNIEnv *env, jclass cls, jlong tensor, jbyteArray src, jfloatArray min,
    jfloatArray max, jint c_out, jint c_in, jfloat threshold, jint layout)
{
  QuantizedTensor *j_tensor = (QuantizedTensor *)tensor;
  jbyte *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  jfloat *jni_min = (*env)->GetPrimitiveArrayCritical(env, min, JNI_FALSE);
  jfloat *jni_max = (*env)->GetPrimitiveArrayCritical(env, max, JNI_FALSE);

  QuantizedFCKernelLoadFromModel(j_tensor, jni_src, jni_min, jni_max, c_out,
                                 c_in, threshold, layout);

  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, min, jni_min, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, max, jni_max, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCDataDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCDataDescInit(JNIEnv *env,
                                                                jclass cls,
                                                                jint batch_size,
                                                                jint channel)
{
  QuantizedTensor *tmp = (QuantizedTensor *)malloc(sizeof(QuantizedTensor));
  QuantizedFCDataDescInit(tmp, batch_size, channel);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCDataInit
 * Signature: (J[FIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCDataInit(
    JNIEnv *env, jclass cls, jlong tensor, jfloatArray src, jint srcOffset,
    jint batch_size, jint channel, jfloat threshold, jint layout)
{
  QuantizedTensor *j_tensor = (QuantizedTensor *)tensor;

  jfloat *jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  QuantizedFCDataInit(j_tensor, jni_src + srcOffset, batch_size, channel,
                      threshold, layout);
  (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    loadRuntime
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT jint JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_loadRuntime(JNIEnv *env,
                                                             jclass cls,
                                                             jstring path)
{
  const char *jPath = (*env)->GetStringUTFChars(env, path, 0);
  jint ret = ManualRuntimeLoadLib(jPath);
  (*env)->ReleaseStringUTFChars(env, path, jPath);
  return ret;
}

#ifdef __cplusplus
}
#endif
