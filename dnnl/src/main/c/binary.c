#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_BinaryDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long src0_desc,
  long src1_desc,
  long dst_desc)
{

  dnnl_binary_desc_t* binary_desc = malloc(sizeof(dnnl_binary_desc_t));

   CHECK(dnnl_binary_desc_init(
     binary_desc,
     (dnnl_alg_kind_t) alg_kind,
     (const dnnl_memory_desc_t*) src0_desc,
     (const dnnl_memory_desc_t*) src1_desc,
     (const dnnl_memory_desc_t*) dst_desc
   )
  );

  return (long)binary_desc;
}

// TODO free the inner product desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeBinaryDescInit(
  JNIEnv *env, jclass cls, jlong binary_desc) {
  free((dnnl_binary_desc_t *) binary_desc);
  return;
}

#ifdef __cplusplus
}
#endif
