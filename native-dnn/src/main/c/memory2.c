#define _POSIX_C_SOURCE 200112L
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define PREFIX(func) Java_com_intel_analytics_bigdl_dnnl_Memory_##func

static void fast_copy(float *dst, float *src, const size_t n,
                      const size_t element_size) {
  int threshold = omp_get_max_threads();
  const int run_parallel = (n >= threshold) && (omp_in_parallel() == 0);

  if (run_parallel) {
    const int block_mem_size = 256 * 1024;
    const int block_size = block_mem_size / element_size;
#pragma omp parallel for
    for (size_t i = 0; i < n; i += block_size)
      memcpy(dst + i, src + i,
             (i + block_size > n) ? (n - i) * element_size : block_mem_size);

    return;
  }

  memcpy(dst, src, element_size * n);
}

static void fast_copy_int(int *dst, int *src, const size_t n,
                          const size_t element_size) {
  int threshold = omp_get_max_threads();
  const int run_parallel = (n >= threshold) && (omp_in_parallel() == 0);

  if (run_parallel) {
    const int block_mem_size = 256 * 1024;
    const int block_size = block_mem_size / element_size;
#pragma omp parallel for
    for (size_t i = 0; i < n; i += block_size)
      memcpy(dst + i, src + i,
             (i + block_size > n) ? (n - i) * element_size : block_mem_size);

    return;
  }

  memcpy(dst, src, element_size * n);
}

static void fast_copy_byte(char *dst, char *src, const size_t n,
                           const size_t element_size) {
  int threshold = omp_get_max_threads();
  const int run_parallel = (n >= threshold) && (omp_in_parallel() == 0);

  if (run_parallel) {
    const int block_mem_size = 256 * 1024;
    const int block_size = block_mem_size / element_size;
#pragma omp parallel for
    for (size_t i = 0; i < n; i += block_size)
      memcpy(dst + i, src + i,
             (i + block_size > n) ? (n - i) * element_size : block_mem_size);

    return;
  }

  memcpy(dst, src, element_size * n);
}

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL PREFIX(SetDataHandle)(JNIEnv *env, jclass cls,
                                              jlong memory, jlong data,
                                              jint offset) {
  float *j_data = (float *)data;
  CHECK(dnnl_memory_set_data_handle((dnnl_memory_t)memory,
                                      j_data + offset));

  return (long)j_data;
}

JNIEXPORT jlong JNICALL PREFIX(Zero)(JNIEnv *env, jclass cls, jlong data,
                                     jint length, jint element_size) {
  memset((float *)data, 0, length * element_size);
  return 0;
}

JNIEXPORT jlong JNICALL PREFIX(CopyPtr2Ptr)(JNIEnv *env, jclass cls, jlong src,
                                            jint srcOffset, jlong dst,
                                            jint dstOffset, jint length,
                                            jint element_size) {
  fast_copy((float *)dst + dstOffset, (float *)src + srcOffset, length, element_size);
  return 0;
}

JNIEXPORT jlong JNICALL PREFIX(CopyArray2Ptr)(JNIEnv *env, jclass cls,
                                              jfloatArray src, jint srcOffset,
                                              jlong dst, jint dstOffset,
                                              jint length, jint element_size) {
  float *j_src = (*env)->GetPrimitiveArrayCritical(env, src, JNI_FALSE);
  float *j_dst = (float *)dst;
  fast_copy(j_dst + dstOffset, j_src + srcOffset, length, element_size);
  (*env)->ReleasePrimitiveArrayCritical(env, src, j_src, 0);
  return 0;
}

JNIEXPORT jlong JNICALL PREFIX(CopyPtr2Array)(JNIEnv *env, jclass cls,
                                              jlong src, jint srcOffset,
                                              jfloatArray dst, jint dstOffset,
                                              jint length, jint element_size) {
  float *j_dst = (*env)->GetPrimitiveArrayCritical(env, dst, JNI_FALSE);
  fast_copy(j_dst + dstOffset, (float *)src + srcOffset, length, element_size);
  (*env)->ReleasePrimitiveArrayCritical(env, dst, j_dst, 0);
  return 0;
}

JNIEXPORT jlong JNICALL PREFIX(AlignedMalloc)(JNIEnv *env, jclass cls,
                                              jint capacity, jint align) {
  void *p;
  // int ret = posix_memalign(&p, align, capacity);
  p = mkl_malloc(capacity, align);
  if (p != NULL) {
    return (long)p;
  } else {
    return (long)0;
  }
}

JNIEXPORT void JNICALL PREFIX(AlignedFree)(JNIEnv *env, jclass cls, jlong ptr) {
  // free((void*)ptr);
  mkl_free((void *)ptr);
}

JNIEXPORT void JNICALL PREFIX(SAdd)(JNIEnv *env, jclass cls, jint n, jlong aPtr,
                                    jint aOffset, jlong bPtr, jint bOffset,
                                    jlong yPtr, jint yOffset) {
  vsAdd(n, (float *)aPtr + aOffset, (float *)bPtr + bOffset,
        (float *)yPtr + yOffset);
}

JNIEXPORT void JNICALL PREFIX(Scale)(JNIEnv *env, jclass cls, jint num,
                                     jfloat scale, jlong x, jlong y) {
  cblas_scopy(num, (float *)x, 1, (float *)y, 1);
  cblas_sscal(num, scale, (float *)y, 1);
}

JNIEXPORT void JNICALL PREFIX(Axpby)(JNIEnv *env, jclass cls, jint n,
                                     jfloat alpha, jlong x, jfloat beta,
                                     jlong y) {
  cblas_saxpby(n, alpha, (float *)x, 1, beta, (float *)y, 1);
}

JNIEXPORT void JNICALL PREFIX(Set)(JNIEnv *env, jclass cls, jlong ptr,
                                   jfloat value, jint length,
                                   jint elementSize) {
  jfloat *data = (jfloat *)ptr;
  for (int i = 0; i < length; i++) {
    data[i] = value;
  }
}

JNIEXPORT jlongArray JNICALL PREFIX(GetShape)(JNIEnv* env,
                                             jclass cls,
                                             jlong desc) {
  dnnl_memory_desc_t* jni_desc = (dnnl_memory_desc_t*)desc;
  int ndims = jni_desc->ndims;
  int64_t* dims = jni_desc->dims;

  jlongArray result = (*env)->NewLongArray(env, ndims);
  (*env)->SetLongArrayRegion(env, result, 0, ndims, dims);
  return result;
}

JNIEXPORT jlongArray JNICALL PREFIX(GetPaddingShape)(JNIEnv* env,
                                                    jclass cls,
                                                    jlong desc) {
  dnnl_memory_desc_t* jni_desc = (dnnl_memory_desc_t*)desc;
  int ndims = jni_desc->ndims;
  int64_t* dims = jni_desc->padded_dims;

  jlongArray result = (*env)->NewLongArray(env, ndims);
  (*env)->SetLongArrayRegion(env, result, 0, ndims, dims);
  return result;
}

JNIEXPORT jint JNICALL PREFIX(GetLayout)(JNIEnv* env, jclass cls, jlong desc) {
  dnnl_memory_desc_t* jni_desc = (dnnl_memory_desc_t*)desc;
  return (int)(jni_desc->format_kind);
}

JNIEXPORT jint JNICALL PREFIX(GetDataType)(JNIEnv* env,
                                           jclass cls,
                                           jlong desc) {
  dnnl_memory_desc_t* jni_desc = (dnnl_memory_desc_t*)desc;
  return (int)(jni_desc->data_type);
}

JNIEXPORT jlong JNICALL PREFIX(CopyPtr2ByteArray)(JNIEnv* env,
                                                  jclass cls,
                                                  jlong src,
                                                  jint srcOffset,
                                                  jbyteArray dst,
                                                  jint dstOffset,
                                                  jint length,
                                                  jint element_size) {
  char* j_dst = (*env)->GetPrimitiveArrayCritical(env, dst, JNI_FALSE);
  fast_copy_byte(j_dst + dstOffset, (char*)src + srcOffset, length, 1);
  (*env)->ReleasePrimitiveArrayCritical(env, dst, j_dst, 0);
  return 0;
}

JNIEXPORT jlong JNICALL PREFIX(CopyPtr2IntArray)(JNIEnv* env,
                                                 jclass cls,
                                                 jlong src,
                                                 jint srcOffset,
                                                 jintArray dst,
                                                 jint dstOffset,
                                                 jint length,
                                                 jint element_size) {
  int* j_dst = (*env)->GetPrimitiveArrayCritical(env, dst, JNI_FALSE);
  fast_copy_int(j_dst + dstOffset, (int*)src + srcOffset, length, 4);
  (*env)->ReleasePrimitiveArrayCritical(env, dst, j_dst, 0);
  return 0;
}

#ifdef __cplusplus
}
#endif
