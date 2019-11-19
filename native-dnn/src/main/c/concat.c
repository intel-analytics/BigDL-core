#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

//dnnl_status_t MKLDNN_API dnnl_concat_primitive_desc_create(
//        dnnl_primitive_desc_t *concat_primitive_desc,
//        const dnnl_memory_desc_t *output_desc, int n, int concat_dimension,
//        const_dnnl_primitive_desc_t *input_pds);

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_ConcatPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long output_desc,
  int n,
  int concat_dimension,
  jlongArray input_pds,
  long attr,
  long engine)
{
  dnnl_primitive_desc_t concat_desc = malloc(sizeof(dnnl_primitive_desc_t));

  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, input_pds, JNI_FALSE);
  dnnl_memory_desc_t *srcs[n];
  for (int i = 0; i < n; i++) {
    srcs[i] = (dnnl_memory_desc_t*)(j_inputs[i]);
  }

   CHECK(dnnl_concat_primitive_desc_create(
     &concat_desc,
     (dnnl_memory_desc_t *)output_desc,
     n,
     concat_dimension,
     srcs[0],
     (const_dnnl_primitive_attr_t)attr,
     (dnnl_engine_t)engine
   )
  );

  (*env)->ReleasePrimitiveArrayCritical(env, input_pds, j_inputs, 0);

  return (long)concat_desc;
}

// TODO free the concat desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeConcatDescInit
(JNIEnv *env, jclass cls, jlong concat_desc)
{
  free((dnnl_primitive_desc_t) concat_desc);
  return;
}

// TODO free the View desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeViewDescInit
(JNIEnv *env, jclass cls, jlong view_primitive_desc)
{
  free((dnnl_primitive_desc_t *) view_primitive_desc);
  return;
}

#ifdef __cplusplus
}
#endif
