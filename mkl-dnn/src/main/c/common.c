#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long op_desc, long engine,
  long hint_forward_primitive_desc)
{
  mkldnn_primitive_desc_t primitive_desc;

  CHECK(mkldnn_primitive_desc_create(
      &primitive_desc,
      (const_mkldnn_op_desc_t)op_desc,
      (mkldnn_engine_t)engine,
      (const_mkldnn_primitive_desc_t)hint_forward_primitive_desc));

  return (long)primitive_desc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescDestroy(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  CHECK(mkldnn_primitive_desc_destroy(
      (mkldnn_primitive_desc_t)primitive_desc));
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreate0(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  mkldnn_primitive_t primitive;

  CHECK(
    mkldnn_primitive_create(
      &primitive,
      (const_mkldnn_primitive_desc_t)primitive_desc,
      NULL,
      NULL));

  return (long)primitive;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreate2(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  jlongArray inputs,
  jintArray indexes,
  jint input_len,
  jlongArray outputs,
  jint output_len)
{
  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  jint *j_indexes = (*env)->GetPrimitiveArrayCritical(env, indexes, JNI_FALSE);

  mkldnn_primitive_t primitive;

  mkldnn_primitive_at_t srcs[input_len];
  const_mkldnn_primitive_t dsts[output_len];
  
  for (int i = 0; i < input_len; i++) {
    srcs[i] = mkldnn_primitive_at(
      (mkldnn_primitive_t)(j_inputs[i]),
      j_indexes[i]);
  }

  for (int i = 0; i < output_len; i++) {
    dsts[i] = (const_mkldnn_primitive_t)(j_outputs[i]);
  }

  CHECK(
    mkldnn_primitive_create(
      &primitive,
      (const_mkldnn_primitive_desc_t)primitive_desc,
      srcs,
      dsts));

  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, indexes, j_indexes, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDestroy(
  JNIEnv *env, jclass cls,
  long primitive)
{
  mkldnn_primitive_destroy((mkldnn_primitive_t)primitive);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreateForSubmit(
  JNIEnv *env, jclass cls,
  long primitive_desc,
  jlongArray inputs, // TODO java array
  int length_inputs,
  jlongArray outputs,
  int length_outputs) // java array
{
  jlong * j_inputs = (*env)->GetPrimitiveArrayCritical(env, inputs, JNI_FALSE);
  jlong * j_outputs = (*env)->GetPrimitiveArrayCritical(env, outputs, JNI_FALSE);
  mkldnn_primitive_t primitive;

  mkldnn_primitive_at_t primitive_at[length_inputs];
  const_mkldnn_primitive_t const_primitive[length_outputs];
  int i = 0;
  while (i < length_inputs) {
    const_mkldnn_primitive_t *temp = (const_mkldnn_primitive_t *)j_inputs[i];
    primitive_at[i] = mkldnn_primitive_at(*temp, 0);
    i ++;
  }
  i = 0;
  while (i < length_outputs) {
    const_mkldnn_primitive_t *temp = (const_mkldnn_primitive_t *)j_outputs[i];
    const_primitive[i] = *temp;
    i ++;
  }

  CHECK(
    mkldnn_primitive_create(
      &primitive,
      (const_mkldnn_primitive_desc_t)primitive_desc,
      primitive_at,
      const_primitive));

  (*env)->ReleasePrimitiveArrayCritical(env, inputs, j_inputs, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputs, j_outputs, 0);

  return (long)primitive;
}

#ifdef __cplusplus
}
#endif
