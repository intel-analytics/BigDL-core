#define _POSIX_C_SOURCE 200112L
#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryDescInit(
  JNIEnv *env, jclass cls,
  int ndims,
  jintArray dims,
  int data_type,
  int format)
{
  jint * j_dims = (*env)->GetPrimitiveArrayCritical(env,
                                                    dims,
                                                    JNI_FALSE);

  mkldnn_memory_desc_t *desc = malloc(sizeof(mkldnn_memory_desc_t));
  CHECK(
    mkldnn_memory_desc_init(desc,
                            ndims,
                            j_dims,
                            (mkldnn_data_type_t)data_type,
                            (mkldnn_memory_format_t)format));

  (*env)->ReleasePrimitiveArrayCritical(env, dims, j_dims, 0);

  return (long)desc;
}

JNIEXPORT long
  JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long memory_desc,
  long engine)
{
  mkldnn_primitive_desc_t primitive_desc;
  CHECK(
    mkldnn_memory_primitive_desc_create(
      &primitive_desc,
      (mkldnn_memory_desc_t *)memory_desc,
      (mkldnn_engine_t)engine)
    );
  return (long)primitive_desc;
}

JNIEXPORT long
JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryGetDataHandle(
  JNIEnv *env, jclass cls,
  long memory)
{
  void *req = NULL;
  CHECK(
    mkldnn_memory_get_data_handle(
      (mkldnn_primitive_t)memory,
      &req));

  return (long)req;
}

JNIEXPORT long JNICALL
Java_com_intel_analytics_bigdl_mkl_MklDnn_MemorySetDataHandle(
  JNIEnv *env, jclass cls,
  long memory, jfloatArray data, jint offset)
{
  float *j_data = (*env)->GetPrimitiveArrayCritical(env, data, JNI_FALSE);

  CHECK(
    mkldnn_memory_set_data_handle(
      (mkldnn_primitive_t)memory,
      j_data + offset));

  return (long)j_data;
}

JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryReleaseDataHandle(
  JNIEnv *env, jclass cls,
  jfloatArray data, long j_data)
{
  (*env)->ReleasePrimitiveArrayCritical(env, data, (float *)j_data, 0);
}


/** Queries primitive descriptor for memory descriptor
 *
 * @returns NULL in case of any error
 */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescQueryMemory(
  JNIEnv *env, jclass cls, long primitive_desc)
{
  const mkldnn_memory_desc_t *qmd = mkldnn_primitive_desc_query_memory_d(
  (const_mkldnn_primitive_desc_t) primitive_desc);

  return (long)qmd;
}

/** Returns the size (in bytes) that is required for given @p
 * memory_primitive_desc */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescGetSize(
  JNIEnv *env, jclass cls, long primitive_desc)
{
  size_t res =  mkldnn_memory_primitive_desc_get_size(
  (const_mkldnn_primitive_desc_t)primitive_desc);
  return (long)res;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    copyFloatBuffer2Array
 * Signature: (Ljava/nio/FloatBuffer;I[FII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_copyFloatBuffer2Array
  (JNIEnv *env, jclass cls, jobject buffer, jint bufferOffset,
   jfloatArray array, jint arrayOffset, jint length)
{
  float *src = (float*)(*env)->GetDirectBufferAddress(env, buffer);
  float *dst = (float*)(*env)->GetPrimitiveArrayCritical(env, array, 0);              
  memcpy(dst + arrayOffset, src + bufferOffset, length * sizeof(float));

  (*env)->ReleasePrimitiveArrayCritical(env, array, dst, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    copyArray2FloatBuffer
 * Signature: (Ljava/nio/FloatBuffer;I[FII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_copyArray2FloatBuffer
  (JNIEnv *env, jclass cls, jobject buffer, jint bufferOffset,
   jfloatArray array, jint arrayOffset, jint length)
{
  float *dst = (float*)(*env)->GetDirectBufferAddress(env, buffer);
  float *src = (float*)(*env)->GetPrimitiveArrayCritical(env, array, 0);              
  memcpy(dst + bufferOffset, src + arrayOffset, length * sizeof(float));

  (*env)->ReleasePrimitiveArrayCritical(env, array, src, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    fillFloatBuffer
 * Signature: (Ljava/nio/FloatBuffer;IFI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_fillFloatBuffer
  (JNIEnv *env, jclass cls, jobject buffer, jint bufferOffset, jfloat value,
   jint length)
{
  float *j_buffer = (float*)(*env)->GetDirectBufferAddress(env, buffer);
  if (value == 0) {
    memset(j_buffer + bufferOffset, value, length * sizeof(float));
  } else {
    int i;
    for (i = 0; i < length; i++) {
      j_buffer[bufferOffset + i] = value;
    }
  }
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemoryGetDataHandleOfArray
 * Signature: ([F)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryGetDataHandleOfArray
  (JNIEnv *env, jclass cls, jfloatArray array)
{
  float *j_data = (*env)->GetPrimitiveArrayCritical(env, array, JNI_FALSE);
  return (jlong)(j_data);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemorySetDataHandleWithBuffer
 * Signature: (JJIILjava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemorySetDataHandleWithBuffer
  (JNIEnv *env, jclass cls, jlong primitive, jlong array, jint offset, jint length, jobject buffer, jint position)
{
  char *j_buffer = (char*)(*env)->GetDirectBufferAddress(env, buffer);
  float *j_array = (float*)array;

  if (array != 0) {
    memcpy(j_buffer + position, j_array + offset, length * sizeof(float));
  }

  CHECK(
    mkldnn_memory_set_data_handle(
      (mkldnn_primitive_t)primitive,
      (float*)(j_buffer + position)));
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemoryAlignedMalloc
 * Signature: (II)L
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryAlignedMalloc
  (JNIEnv *env, jclass cls, jint capacity, jint align)
{
#ifdef WIN32
    return (long)_aligned_malloc(capacity, align);
#else
    void *p;
    return !posix_memalign(&p, align, capacity) ? (long)p : (long)NULL;
#endif
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemorySetDataHandleWithPtr
 * Signature: (JJIIJI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemorySetDataHandleWithPtr
  (JNIEnv *env, jclass cls, jlong primitive, jlong array, jint offset, jint length, jlong buffer, jint position)
{
  char *j_buffer = (char*)buffer;
  float *j_array = (float*)array;

  if (array != 0) {
    memcpy(j_buffer + position, j_array + offset, length * sizeof(float));
  }

  CHECK(
    mkldnn_memory_set_data_handle(
      (mkldnn_primitive_t)primitive,
      (float*)(j_buffer + position)));
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    copyPtr2Array
 * Signature: (JI[FII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_copyPtr2Array
  (JNIEnv *env, jclass cls, jlong ptr, jint position, jfloatArray array, jint offset, jint length)
{
  float *src = (float*)ptr;
  float *dst = (float*)(*env)->GetPrimitiveArrayCritical(env, array, 0);              
  memcpy(dst + offset, src + position, length * sizeof(float));

  (*env)->ReleasePrimitiveArrayCritical(env, array, dst, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemoryAlignedFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryAlignedFree
  (JNIEnv *env, jclass cls, jlong ptr)
{
  free((void*)ptr);
}

#ifdef __cplusplus
}
#endif
