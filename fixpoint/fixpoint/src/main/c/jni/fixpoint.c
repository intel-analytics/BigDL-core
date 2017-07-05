#include <nn-fixpoint.h>
#include <stdio.h>
#include <stdlib.h>

#include "com_intel_analytics_bigdl_fixpoint_FixPoint.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_printHello(JNIEnv *env,
                                                            jclass cls)
{
  printf("Hello FixPoint!");
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelDescInit
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelDescInit(
    JNIEnv *env, jclass cls, jint c_out, jint c_in, jint kernel_h,
    jint kernel_w)
{
  FixTensor *tmp = (FixTensor*)malloc(sizeof(FixTensor));
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelInit
 * Signature: (J[FIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelInit(
    JNIEnv *env, jclass cls, jlong fix_tensor, jfloatArray src, jint srcOffset,
    jint c_out, jint c_in, jint kernel_h, jint kernel_w, jfloat threshold,
    jint layout)
{
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelLoadFromModel
 * Signature: ([CI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelLoadFromModel(
    JNIEnv *env, jclass cls, jcharArray src,
    jint srcOffset, jfloatArray min, jfloatArray max, jint c_out, jint c_in,
    jint kernel_h, jint kernel_w, jfloat threshold, jint layout)
{
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvDataDescInit
 * Signature: (IIIIIIIIIIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvDataDescInit(
    JNIEnv *env, jclass cls, jint c_in, jint kernel_h, jint kernel_w,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint dilation_h,
    jint dilation_w, jint batch_size, jint h_in, jint w_in)
{
  FixTensor *tmp = (FixTensor*)malloc(sizeof(FixTensor));
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvDataInit
 * Signature: (J[FIIIIIIIIIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvDataInit(
    JNIEnv *env, jclass cls, jlong fix_tensor, jfloatArray src, jint srcOffset,
    jint c_in, jint kernel_h, jint kernel_w, jint stride_h, jint stride_w,
    jint pad_h, jint pad_w, jint dilation_h, jint dilation_w, jint batch_size,
    jint h_in, jint w_in, jfloat threshold, jint layout)
{
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelSumDescInit
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelSumDescInit(
    JNIEnv *env, jclass cls, jint c_out)
{
  FixTensor *tmp = (FixTensor*)malloc(sizeof(FixTensor));
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelSumInit(
    JNIEnv *env, jclass cls, jlong fix_tensor, jfloatArray src, jint srcOffset,
    jint n, jint c, jint h, jint w)
{
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    InternalMixPrecisionConvolutionGEMM
 * Signature: (IJJ[FIIIIJ[FIIIIIF)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_InternalMixPrecisionConvolutionGEMM(
    JNIEnv *env, jclass cls, jint layout, jlong pa, jlong pb, jfloatArray pc,
    jint pcOffset, jint m, jint n, jint k, jlong kernel_sum, jfloatArray bias,
    jint biasOffset, jint batch_size, jint channel_per_group, jint height_out,
    jint width_out, jfloat fault_tolerance)
{
}

#ifdef __cplusplus
}
#endif
