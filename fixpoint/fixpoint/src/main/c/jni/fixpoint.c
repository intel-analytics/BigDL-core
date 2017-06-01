#include <nn-fixpoint.h>
#include <stdio.h>

#include "com_intel_analytics_bigdl_fixpoint_FixPoint.h"

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_printHello
  (JNIEnv *env, jclass cls)
{
    printf("Hello FixPoint!");
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpCreate
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpCreate
  (JNIEnv *env, jclass cls, jint type)
{
    /* TODO only support NCHW */
    FixConvOpDesc *desc = FixConvOpCreate(NCHW);
    return (long)desc;
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpSetupConvParameter
 * Signature: (JJJJJJJJJJJJ[FJZ[FJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpSetupConvParameter
  (JNIEnv *env, jclass cls, jlong desc, jlong channel_out, jlong channel_in,
   jlong group, jlong kernel_h, jlong kernel_w, jlong stride_h, jlong stride_w,
   jlong dilation_h, jlong dilation_w, jlong pad_h, jlong pad_w,
   jfloatArray weight, jlong weightOffset, jboolean with_bias, jfloatArray bias, jlong biasOffset, jboolean relu_fusion)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    jfloat* jWeight = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, weight, 0));
    jfloat* jBias = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, bias, 0));

    FixConvOpSetupConvParameter(jDesc, channel_out, channel_in, group, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, jWeight+weightOffset, true, jBias+biasOffset, false);

    (*env)->ReleasePrimitiveArrayCritical(env, weight, jWeight, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, bias, jBias, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpQuantizeKernel
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpQuantizeKernel
  (JNIEnv *env, jclass cls, jlong desc, jfloat threshold)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    FixConvOpQuantizeKernel(jDesc, threshold);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpQuantizeData
 * Signature: (JJJJJ[FJF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpQuantizeData
  (JNIEnv *env, jclass cls, jlong desc, jlong batch_size, jlong channels, jlong height_in, jlong width_in, jfloatArray src, jlong srcOffset, jfloat sw_threshold)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    jfloat* jSrc = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, src, 0));

    FixConvOpQuantizeData(jDesc, batch_size, channels, height_in, width_in, jSrc+srcOffset, sw_threshold);

    (*env)->ReleasePrimitiveArrayCritical(env, src, jSrc, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpSetupTargetBuffer
 * Signature: (J[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpSetupTargetBuffer
  (JNIEnv *env, jclass cls, jlong desc, jfloatArray dst, jlong dstOffset)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    jfloat* jDst = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, dst, 0));

    /*
    printf("%d jDesc = %x\n", __LINE__, jDesc);
    printf("%d jDesc = %x\n", __LINE__, jDst);
    */

    FixConvOpSetupTargetBuffer(jDesc, jDst+dstOffset);

    (*env)->ReleasePrimitiveArrayCritical(env, dst, jDst, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpExecute
 * Signature: (JF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpExecute
  (JNIEnv *env, jclass cls, jlong desc, jfloat fault_tolerance)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    FixConvOpExecute(jDesc, fault_tolerance);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpFree
  (JNIEnv *env, jclass cls, jlong desc)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    FixConvOpFree(jDesc);
}

/*
 * Class:     com_intel_analytics_bigdl_fixpoint_FixPoint
 * Method:    FixConvOpExecuteAll
 * Signature: (JJJJJ[FJF[FJF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_fixpoint_FixPoint_FixConvOpExecuteAll
  (JNIEnv *env, jclass cls,
   jlong desc,
   jlong batch_size,
   jlong channels,
   jlong height_in,
   jlong width_in,
   jfloatArray src,
   jlong srcOffset,
   jfloat sw_threshold,
   jfloatArray dst,
   jlong dstOffset,
   jfloat fault_tolerance)
{
    FixConvOpDesc *jDesc = (FixConvOpDesc*)desc;
    jfloat* jSrc = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, src, 0));
    jfloat* jDst = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, dst, 0));

    FixConvOpQuantizeData(jDesc, batch_size, channels, height_in, width_in, jSrc+srcOffset, sw_threshold);
    FixConvOpSetupTargetBuffer(jDesc, jDst+dstOffset);
    FixConvOpExecute(jDesc, fault_tolerance);

    (*env)->ReleasePrimitiveArrayCritical(env, dst, jDst, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, src, jSrc, 0);
}
