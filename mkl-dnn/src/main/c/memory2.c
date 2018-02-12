#define _POSIX_C_SOURCE 200112L
#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_Memory.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    SetDataHandle
 * Signature: (JJI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_SetDataHandle
  (JNIEnv *env, jclass cls, jlong primitive, jlong data, jint offset)
  {
    float *j_data = (float*)data;
    CHECK(
      mkldnn_memory_set_data_handle(
        (mkldnn_primitive_t)primitive,
        j_data + offset));

    return (long)j_data;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    Zero
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_Zero
  (JNIEnv *env, jclass cls, jlong data, jint length, jint element_size)
  {
    memset((float*)data, 0, length * element_size);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyPtr2Ptr
 * Signature: (JIJIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyPtr2Ptr
  (JNIEnv *env, jclass cls, jlong src, jint srcOffset,
   jlong dst, jint dstOffset, jint length, jint element_size)
  {
    memcpy((float*)dst + dstOffset, (float*)src + srcOffset, length * element_size);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyArray2Ptr
 * Signature: ([FIJIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyArray2Ptr
  (JNIEnv *env, jclass cls, jfloatArray src, jint srcOffset,
   jlong dst, jint dstOffset, jint length, jint element_size)
  {
    float *j_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
    float *j_dst = (float*)dst;
    memcpy(j_dst + dstOffset, j_src + srcOffset, length * element_size);
    (*env)->ReleasePrimitiveArrayCritical(env, src, j_src, 0);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    CopyPtr2Array
 * Signature: (JI[FIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_CopyPtr2Array
  (JNIEnv *env, jclass cls, jlong src, jint srcOffset,
   jfloatArray dst, jint dstOffset, jint length, jint element_size)
  {
    float *j_dst = (*env)->GetPrimitiveArrayCritical(env, dst, JNI_FALSE);
    // float *j_src = (float *)src;
    // int i = 0;
    // for (i = 0; i < 10; i++) {
    //   printf("%f\n", j_src[i + srcOffset]);
    // }
    // fflush(stdout);
    memcpy(j_dst + dstOffset, (float *)src + srcOffset, length * element_size);
    (*env)->ReleasePrimitiveArrayCritical(env, dst, j_dst, 0);
    return 0;
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    AlignedMalloc
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_AlignedMalloc
  (JNIEnv *env, jclass cls, jint capacity, jint align)
  {
    void *p;
    int ret = posix_memalign(&p, align, capacity);
    if (!ret) {
      return (long)p;
    } else {
      return (long)0;
    }
  }

/*
 * Class:     com_intel_analytics_bigdl_mkl_Memory
 * Method:    AlignedFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_Memory_AlignedFree
  (JNIEnv *env, jclass cls, jlong ptr)
  {
    free((void*)ptr);
  }
#ifdef __cplusplus
}
#endif
