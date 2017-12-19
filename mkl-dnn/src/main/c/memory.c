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
                            (mkldnn_memory_format_t)format)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, dims, j_dims, 0);

  return (long)desc;
}

JNIEXPORT long
  JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long memory_desc,
  int engine)
{
  mkldnn_primitive_desc_t *primitive_desc =
    malloc(sizeof(mkldnn_primitive_desc_t));
  CHECK(
    mkldnn_memory_primitive_desc_create(
      primitive_desc,
      (const mkldnn_memory_desc_t *)memory_desc,
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
  mkldnn_primitive_t *j_memory = (mkldnn_primitive_t*)(memory);
  CHECK(
    mkldnn_memory_get_data_handle(*j_memory, &req)
    );

  return (long)req;
}

// TODO data should be java array
JNIEXPORT void JNICALL
Java_com_intel_analytics_bigdl_mkl_MklDnn_MemorySetDataHandle(
  JNIEnv *env, jclass cls,
  long memory, long data)
{
  float *j_data = (float*) data;
  mkldnn_primitive_t *j_memory = (mkldnn_primitive_t*)(memory);

  CHECK(
    mkldnn_memory_set_data_handle(*j_memory, j_data)
    );
}
#ifdef __cplusplus
}
#endif
