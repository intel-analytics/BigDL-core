/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_analytics_bigdl_bigquant_BigQuant */

#ifndef _Included_com_intel_analytics_bigdl_bigquant_BigQuant
#define _Included_com_intel_analytics_bigdl_bigquant_BigQuant
#ifdef __cplusplus
extern "C" {
#endif
#undef com_intel_analytics_bigdl_bigquant_BigQuant_NCHW
#define com_intel_analytics_bigdl_bigquant_BigQuant_NCHW 0L
#undef com_intel_analytics_bigdl_bigquant_BigQuant_NHWC
#define com_intel_analytics_bigdl_bigquant_BigQuant_NHWC 1L
/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    printHello
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_printHello(JNIEnv *, jclass);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    loadRuntime
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT jint JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_loadRuntime(JNIEnv *, jclass,
                                                             jstring);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelDescInit
 * Signature: (IIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelDescInit(JNIEnv *,
                                                                    jclass,
                                                                    jint, jint,
                                                                    jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelInit
 * Signature: (J[FIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelInit(
    JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jfloat,
    jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelLoadFromModel
 * Signature: (J[BI[F[FIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelLoadFromModel(
    JNIEnv *, jclass, jlong, jbyteArray, jint, jfloatArray, jfloatArray, jint,
    jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvDataDescInit
 * Signature: (IIIIIIIIIIII)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvDataDescInit(
    JNIEnv *, jclass, jint, jint, jint, jint, jint, jint, jint, jint, jint,
    jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvDataInit
 * Signature: (J[FIIIIIIIIIIIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvDataInit(
    JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint, jint,
    jint, jint, jint, jint, jint, jint, jint, jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelSumDescInit
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelSumDescInit(JNIEnv *,
                                                                       jclass,
                                                                       jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    ConvKernelSumInit
 * Signature: (J[FIIIII)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_ConvKernelSumInit(
    JNIEnv *, jclass, jlong, jfloatArray, jint, jint, jint, jint, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    MixPrecisionGEMM
 * Signature: (IJJ[FI[FI[FIIIIIF)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_MixPrecisionGEMM(
    JNIEnv *, jclass, jint, jlong, jlong, jfloatArray, jint, jfloatArray, jint,
    jfloatArray, jint, jint, jint, jint, jint, jfloat);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FreeMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FreeMemory(JNIEnv *, jclass,
                                                            jlong);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCKernelDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCKernelDescInit(JNIEnv *,
                                                                  jclass, jint,
                                                                  jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCKernelLoadFromModel
 * Signature: (J[B[F[FIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCKernelLoadFromModel(
    JNIEnv *, jclass, jlong, jbyteArray, jfloatArray, jfloatArray, jint, jint,
    jfloat, jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCDataDescInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCDataDescInit(JNIEnv *,
                                                                jclass, jint,
                                                                jint);

/*
 * Class:     com_intel_analytics_bigdl_bigquant_BigQuant
 * Method:    FCDataInit
 * Signature: (J[FIIIFI)V
 */
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_bigquant_BigQuant_FCDataInit(JNIEnv *, jclass,
                                                            jlong, jfloatArray,
                                                            jint, jint, jint,
                                                            jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
