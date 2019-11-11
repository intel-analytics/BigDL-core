#define _POSIX_C_SOURCE 200112L
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <mkl.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryDescInit(
  JNIEnv *env, jclass cls,
  int ndims,
  jlongArray dims,
  int data_type,
  int format)
{
  jlong * j_dims = (*env)->GetPrimitiveArrayCritical(env,
                                                    dims,
                                                    JNI_FALSE);

  dnnl_memory_desc_t *desc = malloc(sizeof(dnnl_memory_desc_t));
  CHECK_EXCEPTION(env,
                  dnnl_memory_desc_init_by_tag(desc,
                                          ndims,
                                          j_dims,
                                          (dnnl_data_type_t)data_type,
                                          (dnnl_format_tag_t)format));

  (*env)->ReleasePrimitiveArrayCritical(env, dims, j_dims, 0);

  return (long)desc;
}

// TODO free the memory desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeMemoryDescInit
(JNIEnv *env, jclass cls, jlong memory_desc)
{
  free((dnnl_memory_desc_t *) memory_desc);
  return;
}

JNIEXPORT long
  JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryCreate(
  JNIEnv *env, jclass cls,
  long memory_desc,
  long engine)
{
  dnnl_memory_t memory;
  CHECK(
    dnnl_memory_create(
      &memory,
      (dnnl_memory_desc_t *)memory_desc,
      (dnnl_engine_t)engine,
      DNNL_MEMORY_NONE)
    );
  return (long)memory;
}

JNIEXPORT long
JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryGetDataHandle(
  JNIEnv *env, jclass cls,
  long memory)
{
  void *req = NULL;
  CHECK(
    dnnl_memory_get_data_handle(
      (dnnl_memory_t)memory,
      &req));

  return (long)req;
}

JNIEXPORT long JNICALL
Java_com_intel_analytics_bigdl_dnnl_DNNL_MemorySetDataHandle(
  JNIEnv *env, jclass cls,
  long memory, jfloatArray data, jint offset)
{
  float *j_data = (*env)->GetPrimitiveArrayCritical(env, data, JNI_FALSE);

  CHECK(
    dnnl_memory_set_data_handle(
      (dnnl_memory_t)memory,
      j_data + offset));

  return (long)j_data;
}

JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryReleaseDataHandle(
  JNIEnv *env, jclass cls,
  jfloatArray data, long j_data)
{
  (*env)->ReleasePrimitiveArrayCritical(env, data, (float *)j_data, 0);
}


/** Queries primitive descriptor for memory descriptor
 *
 * @returns NULL in case of any error
 */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_PrimitiveDescQueryMemory(
  JNIEnv *env, jclass cls, long primitive_desc)
{
  const dnnl_memory_desc_t *qmd = malloc(sizeof(dnnl_memory_desc_t));
  dnnl_memory_get_memory_desc(
		  (const_dnnl_memory_t) primitive_desc, &qmd);

  return (long)qmd;
}

/** Returns the size (in bytes) that is required for given @p
 * memory_primitive_desc */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryDescGetSize(
  JNIEnv *env, jclass cls, long memory_desc)
{
  size_t res =  dnnl_memory_desc_get_size(
  (const dnnl_memory_desc_t *)memory_desc);
  return (long)res;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    copyFloatBuffer2Array
 * Signature: (Ljava/nio/FloatBuffer;I[FII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_copyFloatBuffer2Array
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
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_copyArray2FloatBuffer
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
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_fillFloatBuffer
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
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryGetDataHandleOfArray
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
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemorySetDataHandleWithBuffer
  (JNIEnv *env, jclass cls, jlong memory, jlong array, jint offset, jint length, jobject buffer, jint position)
{
  char *j_buffer = (char*)(*env)->GetDirectBufferAddress(env, buffer);
  float *j_array = (float*)array;

  if (array != 0) {
    memcpy(j_buffer + position, j_array + offset, length * sizeof(float));
  }

  CHECK(
    dnnl_memory_set_data_handle(
      (dnnl_memory_t)memory,
      (float*)(j_buffer + position)));
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemoryAlignedMalloc
 * Signature: (II)L
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryAlignedMalloc
  (JNIEnv *env, jclass cls, jint capacity, jint align)
{
#ifdef WIN32
    return (long)_aligned_malloc(capacity, align);
#else
    void *p;
    // int ret = posix_memalign(&p, align, capacity);
    p = mkl_malloc(capacity, align);
    if (p != NULL) {
      return (long)p;
    } else {
      return (long)0;
    }
#endif
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    MemorySetDataHandleWithPtr
 * Signature: (JJIIJI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemorySetDataHandleWithPtr
  (JNIEnv *env, jclass cls, jlong primitive, jlong array, jint offset, jint length, jlong buffer, jint position)
{
  char *j_buffer = (char*)buffer;
  float *j_array = (float*)array;

  if (array != 0) {
    memcpy(j_buffer + position, j_array + offset, length * sizeof(float));
  }

  CHECK(
    dnnl_memory_set_data_handle(
      (dnnl_memory_t)primitive,
      (float*)(j_buffer + position)));
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    copyPtr2Array
 * Signature: (JI[FII)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_copyPtr2Array
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
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_MemoryAlignedFree
  (JNIEnv *env, jclass cls, jlong ptr)
{
  // free((void*)ptr);
  mkl_free((void*)ptr);
}

#ifdef __cplusplus
}
#endif
