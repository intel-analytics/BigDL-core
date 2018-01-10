#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

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


#ifdef __cplusplus
}
#endif
