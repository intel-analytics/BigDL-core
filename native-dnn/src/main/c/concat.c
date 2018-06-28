#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

//mkldnn_status_t MKLDNN_API mkldnn_concat_primitive_desc_create(
//        mkldnn_primitive_desc_t *concat_primitive_desc,
//        const mkldnn_memory_desc_t *output_desc, int n, int concat_dimension,
//        const_mkldnn_primitive_desc_t *input_pds);

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConcatPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long output_desc,
  int n,
  int concat_dimension,
  jlongArray input_pds)
{
  mkldnn_primitive_desc_t concat_desc = malloc(sizeof(mkldnn_primitive_desc_t));

  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, input_pds, JNI_FALSE);
  const_mkldnn_primitive_desc_t srcs[n];
  for (int i = 0; i < n; i++) {
    srcs[i] = (const_mkldnn_primitive_desc_t)(j_inputs[i]);
  }

   CHECK(mkldnn_concat_primitive_desc_create(
     &concat_desc,
     (mkldnn_memory_desc_t *)output_desc,
     n,
     concat_dimension,
     srcs)
  );

  (*env)->ReleasePrimitiveArrayCritical(env, input_pds, j_inputs, 0);

  return (long)concat_desc;
}


/** Creates a @p view_primitive_desc for a given @p memory_primitive_desc, with
 * @p dims sizes and @p offset offsets. May fail if layout used does not allow
 * obtain desired view. In this case consider using extract primitive */
//mkldnn_status_t MKLDNN_API mkldnn_view_primitive_desc_create(
//        mkldnn_primitive_desc_t *view_primitive_desc,
//        const_mkldnn_primitive_desc_t memory_primitive_desc,
//        const mkldnn_dims_t dims, const mkldnn_dims_t offsets);

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ViewPrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long memory_primitive_desc,
  jintArray dims,
  jintArray offsets)
{
  mkldnn_primitive_desc_t view_primitive_desc = malloc(sizeof(mkldnn_primitive_desc_t));

  int *j_dims = (*env)->GetPrimitiveArrayCritical(env, dims, JNI_FALSE);
  int *j_offsets = (*env)->GetPrimitiveArrayCritical(env, offsets, JNI_FALSE);

  CHECK(mkldnn_view_primitive_desc_create(
     &view_primitive_desc,
     (const_mkldnn_primitive_desc_t )memory_primitive_desc,
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
  free((mkldnn_primitive_desc_t) concat_desc);
  return;
}

// TODO free the View desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeViewDescInit
(JNIEnv *env, jclass cls, jlong view_primitive_desc)
{
  free((mkldnn_primitive_desc_t *) view_primitive_desc);
  return;
}

#ifdef __cplusplus
}
#endif
