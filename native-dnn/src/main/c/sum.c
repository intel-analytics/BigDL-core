#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_SumPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long output_desc,
  int n,
  jfloatArray scales,
  jlongArray input_pds,
  long attr,
  long engine)
{
  dnnl_primitive_desc_t sum_primitive_desc = malloc(sizeof(dnnl_primitive_desc_t));

  float *j_scales = (*env)->GetPrimitiveArrayCritical(env, scales, JNI_FALSE);
  long *j_input_pds = (*env)->GetPrimitiveArrayCritical(env, input_pds, JNI_FALSE);

  dnnl_memory_desc_t *pds[n];
    for (int i = 0; i < n; i++) {
      pds[i] = (dnnl_memory_desc_t *)(j_input_pds[i]);
    }

  CHECK(dnnl_sum_primitive_desc_create(
     &sum_primitive_desc,
     (dnnl_memory_desc_t*) output_desc,
     n,
     j_scales,
     pds[0],
     (const_dnnl_primitive_attr_t)attr,
     (dnnl_engine_t)engine)
  );

  (*env)->ReleasePrimitiveArrayCritical(env, scales, j_scales, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, input_pds, j_input_pds, 0);

  return (long)sum_primitive_desc;
}

#ifdef __cplusplus
}
#endif
