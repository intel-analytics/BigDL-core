#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

//dnnl_status_t MKLDNN_API dnnl_concat_primitive_desc_create(
//        dnnl_primitive_desc_t *concat_primitive_desc,
//        const dnnl_memory_desc_t *output_desc, int n, int concat_dimension,
//        const_dnnl_primitive_desc_t *input_pds);

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConcatPrimitiveDescCreate(
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
  const_dnnl_primitive_desc_t srcs[n];
  for (int i = 0; i < n; i++) {
    srcs[i] = (const_dnnl_primitive_desc_t)(j_inputs[i]);
  }

   CHECK(dnnl_concat_primitive_desc_create(
     &concat_desc,
     (dnnl_memory_desc_t *)output_desc,
     n,
     concat_dimension,
     srcs,
     (const_dnnl_primitive_attr_t)attr,
     (dnnl_engine_t)engine
   )
  );

  (*env)->ReleasePrimitiveArrayCritical(env, input_pds, j_inputs, 0);

  return (long)concat_desc;
}


/** Creates a @p view_primitive_desc for a given @p memory_primitive_desc, with
 * @p dims sizes and @p offset offsets. May fail if layout used does not allow
 * obtain desired view. In this case consider using extract primitive */
//dnnl_status_t MKLDNN_API dnnl_view_primitive_desc_create(
//        dnnl_primitive_desc_t *view_primitive_desc,
//        const_dnnl_primitive_desc_t memory_primitive_desc,
//        const dnnl_dims_t dims, const dnnl_dims_t offsets);

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ViewPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long memory_primitive_desc,
  jintArray dims,
  jintArray offsets)
{
  dnnl_primitive_desc_t view_primitive_desc = malloc(sizeof(dnnl_primitive_desc_t));

  int *j_dims = (*env)->GetPrimitiveArrayCritical(env, dims, JNI_FALSE);
  int *j_offsets = (*env)->GetPrimitiveArrayCritical(env, offsets, JNI_FALSE);

  CHECK(dnnl_view_primitive_desc_create(
     &view_primitive_desc,
     (const_dnnl_primitive_desc_t )memory_primitive_desc,
     j_dims,
     j_offsets)
  );

  (*env)->ReleasePrimitiveArrayCritical(env, dims, j_dims, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, offsets, j_offsets, 0);

  return (long)view_primitive_desc;
}

// TODO free the concat desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeConcatDescInit
(JNIEnv *env, jclass cls, jlong concat_desc)
{
  free((dnnl_primitive_desc_t) concat_desc);
  return;
}

// TODO free the View desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeViewDescInit
(JNIEnv *env, jclass cls, jlong view_primitive_desc)
{
  free((dnnl_primitive_desc_t *) view_primitive_desc);
  return;
}

#ifdef __cplusplus
}
#endif
