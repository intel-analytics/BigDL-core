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
  FixConvKernelDescInit(tmp, c_out, c_in, kernel_h, kernel_w);
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
    FixTensor *j_fix_tensor = (FixTensor*)fix_tensor;

    jfloat * jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    FixConvKernelInit(j_fix_tensor, jni_src + srcOffset, c_out, c_in, kernel_h , kernel_w ,threshold, layout);
    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelLoadFromModel
 * Signature: (J[BI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelLoadFromModel(
    JNIEnv *env, jclass cls, jlong fix_tensor, jbyteArray src,
    jint srcOffset, jfloatArray min, jfloatArray max, jint c_out, jint c_in,
    jint kernel_h, jint kernel_w, jfloat threshold, jint layout)
{
    FixTensor *j_fix_tensor = (FixTensor*)fix_tensor;
    jbyte* jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    jfloat* jni_min = (*env)->GetPrimitiveArrayCritical(env, min, JNI_FALSE);
    jfloat* jni_max = (*env)->GetPrimitiveArrayCritical(env, max, JNI_FALSE);
    
    FixConvKernelLoadFromModel(j_fix_tensor, jni_src, jni_min, jni_max, c_out, c_in, kernel_h, kernel_w, threshold, layout);

    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, min, jni_min, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, max, jni_max, 0);
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
  FixConvDataDescInit(tmp, c_in, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, batch_size, h_in, w_in);
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
    FixTensor *j_fix_tensor = (FixTensor*)fix_tensor;

    jfloat * jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    FixConvDataInit(j_fix_tensor, jni_src + srcOffset, c_in, kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, batch_size, h_in, w_in, threshold, layout);
    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
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
  FPTensor *tmp = (FPTensor*)malloc(sizeof(FPTensor));
  FixConvKernelSumDescInit(tmp, c_out);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvKernelSumInit(
    JNIEnv *env, jclass cls, jlong fp_tensor, jfloatArray src, jint srcOffset,
    jint n, jint c, jint h, jint w)
{
    FPTensor *j_fp_tensor = (FPTensor*)fp_tensor;

    jfloat * jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    FixConvKernelSumInit(j_fp_tensor, jni_src + srcOffset, n, c, h, w);
    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    InternalMixPrecisionConvolutionGEMM
 * Signature: (IJIJ[FIIII[FI[FIIIIIIF)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_fixpoint_FixPoint_InternalMixPrecisionConvolutionGEMM(
    JNIEnv *env, jclass cls, jint layout, jlong pa, jint paOffset, jint id, jlong pb, jfloatArray pc,
    jint pcOffset, jint m, jint n, jint k, jfloatArray kernel_sum, jint kernel_sum_offset, jfloatArray bias,
    jint biasOffset, jint batch_size, jint channel_per_group, jint height_out,
    jint width_out, jint group, jfloat fault_tolerance)
{
    FixTensor *jni_pa = (FixTensor*)pa;
    FixTensor *jni_pb = (FixTensor*)pb;

    jfloat *jni_pc = (*env)->GetPrimitiveArrayCritical(env, pc, JNI_FALSE);
    jfloat *jni_bias = (*env)->GetPrimitiveArrayCritical(env, bias, JNI_FALSE);
    jfloat *jni_kernel_sum = (*env)->GetPrimitiveArrayCritical(env, kernel_sum, JNI_FALSE);

    printf("jni_pa->shape[0] = %d, jni_pb->shape[0] = %d, jni_pb->shape[1] = %d\n",
           jni_pa->shape[0], jni_pb->shape[0], jni_pb->shape[1]);
    printf("jni_pa->ori_shape[0] = %d, jni_pb->ori_shape[0] = %d\n",
           jni_pa->ori_shape[0], jni_pb->ori_shape[0]);
    printf("weight_offset = %d\n", paOffset);
    printf("ratio_offset = %d\n", jni_pa->shape[0] / group * id);
    printf("kernel_sum_offset = %d\n", kernel_sum_offset);
    printf("bias_offset = %d\n", biasOffset);

    InternalMixPrecisionGEMM(layout, jni_pa->data + paOffset, jni_pb->data, jni_pc + pcOffset,
            jni_pa->shape[0] / group, jni_pb->shape[0], jni_pb->shape[1],
            jni_pa->ratio + jni_pa->shape[0] / group * id, jni_pb->ratio,
            jni_kernel_sum + kernel_sum_offset, jni_pb->min,
            jni_bias + biasOffset, batch_size,
            channel_per_group, height_out, width_out, fault_tolerance,
            jni_pa->shape[0] - jni_pa->ori_shape[0],
            jni_pb->shape[0] - jni_pb->ori_shape[0]);

    (*env)->ReleasePrimitiveArrayCritical(env, pc, jni_pc, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, bias, jni_bias, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, kernel_sum, jni_kernel_sum, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FreeMemory
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FreeMemory(
    JNIEnv *env, jclass cls, jlong ptr)
{
    FixTensor *jni_ptr = (FixTensor*)ptr;

    FreeMemory(jni_ptr->data);
    FreeMemory(jni_ptr->min);
    FreeMemory(jni_ptr->max);
    FreeMemory(jni_ptr->ratio);

    free(jni_ptr);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixFCKernelDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixFCKernelDescInit
  (JNIEnv *env, jclass cls, jint c_out, jint c_in)
{
  FixTensor *tmp = (FixTensor*)malloc(sizeof(FixTensor));
  FixFCKernelDescInit(tmp, c_out, c_in);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixFCKernelLoadFromModel
 * Signature: (J[B[F[FIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixFCKernelLoadFromModel
  (JNIEnv *env,
 jclass cls,
 jlong fix_tensor,
 jbyteArray src,
 jfloatArray min,
 jfloatArray max,
 jint c_out,
 jint c_in,
 jfloat threshold,
 jint layout)
{
    FixTensor *j_fix_tensor = (FixTensor*)fix_tensor;
    jbyte* jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    jfloat* jni_min = (*env)->GetPrimitiveArrayCritical(env, min, JNI_FALSE);
    jfloat* jni_max = (*env)->GetPrimitiveArrayCritical(env, max, JNI_FALSE);
    
    FixFCKernelLoadFromModel(j_fix_tensor, jni_src, jni_min, jni_max, c_out, c_in, threshold, layout);

    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, min, jni_min, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, max, jni_max, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixFCDataDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixFCDataDescInit
  (JNIEnv *env,
 jclass cls,
 jint batch_size,
 jint channel)
{
  FixTensor *tmp = (FixTensor*)malloc(sizeof(FixTensor));
  FixFCDataDescInit(tmp, batch_size, channel);
  return (jlong)tmp;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixFCDataInit
 * Signature: (J[FIIIFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixFCDataInit
  (JNIEnv *env,
 jclass cls,
 jlong fix_tensor,
 jfloatArray src,
 jint srcOffset,
 jint batch_size,
 jint channel,
 jfloat threshold,
 jint layout)
{
    FixTensor *j_fix_tensor = (FixTensor*)fix_tensor;

    jfloat * jni_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    FixFCDataInit(j_fix_tensor, jni_src + srcOffset,
            batch_size, channel, threshold, layout);
    (*env)->ReleasePrimitiveArrayCritical(env, src, jni_src, 0);
}

#ifdef __cplusplus
}
#endif
