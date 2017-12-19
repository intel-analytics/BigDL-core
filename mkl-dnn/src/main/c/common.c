#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long op_desc, int engine,
  long hint_forward_primitive_desc)
{
  mkldnn_primitive_desc_t *primitive_desc =
    malloc(sizeof(mkldnn_primitive_desc_t));

  CHECK(
    mkldnn_primitive_desc_create(
      primitive_desc,
      (const_mkldnn_op_desc_t)op_desc,
      (mkldnn_engine_t)engine,
      (const_mkldnn_primitive_desc_t)hint_forward_primitive_desc));

  return (long)primitive_desc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescDestroy(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  mkldnn_primitive_desc_t *j_primitive_desc =
    (mkldnn_primitive_desc_t *)primitive_desc;

  CHECK(
    mkldnn_primitive_desc_destroy(*j_primitive_desc));
  free(j_primitive_desc);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreate(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  jlongArray inputs, // TODO java array
  jlongArray outputs) // java array
{
  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  mkldnn_primitive_t *primitive = malloc(sizeof(mkldnn_primitive_t));

  CHECK(
    mkldnn_primitive_create(
      primitive,
      *((const_mkldnn_primitive_desc_t *)primitive_desc),
      (mkldnn_primitive_at_t *)j_inputs,
      (const_mkldnn_primitive_t *)j_outputs));

  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDestroy(
  JNIEnv *env, jclass cls,
  long primitive)
{
  mkldnn_primitive_t *j_primitive = (mkldnn_primitive_t *)primitive;
  mkldnn_primitive_destroy(*j_primitive);
  free(j_primitive);
}

#ifdef __cplusplus
}
#endif
