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

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ReorderPrimitiveDescCreate(
  JNIEnv *env, jclass cls, long input, long output) {

     mkldnn_primitive_desc_t reorder_primitive_desc;

     CHECK(
       mkldnn_reorder_primitive_desc_create(
         &reorder_primitive_desc,
         (const_mkldnn_primitive_desc_t)input,
         (const_mkldnn_primitive_desc_t)output)
     );
     return (long)reorder_primitive_desc;
  }

/** Compares two descriptors of memory primitives.
  * @return 1 if the descriptors are the same.
  * @return 0 if the descriptors are different.
  *
  * Use this function to identify whether a reorder is required for the memory
  * primitives.
  */
JNIEXPORT int JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_MemoryPrimitiveDescEqual(
JNIEnv *env, jclass cls, long lhs, long rhs) {
   return mkldnn_memory_primitive_desc_equal(
       (const_mkldnn_primitive_desc_t)lhs,
       (const_mkldnn_primitive_desc_t)rhs);
}

/** Retrieves a reference to the @p primitive_desc descriptor of given @p
  * primitive.
  *
  * @warning
  *     Returned object must not be destroyed by user. 'const' qualifier of the
  *     returned object prevents such attempts.
  */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveGetPrimitiveDesc(
  JNIEnv *env, jclass cls, long primitive) {

     const_mkldnn_primitive_desc_t primitive_desc;
     CHECK(mkldnn_primitive_get_primitive_desc((const_mkldnn_primitive_t)primitive, &primitive_desc));
     return (long)primitive_desc;
  }

/** Queries primitive descriptor for primitive descriptor
  *
  * @returns NULL in case of any error
  */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescQueryPd(
JNIEnv *env, jclass cls, long primitive, int what, int index) {

  const_mkldnn_primitive_desc_t pd;
  pd = mkldnn_primitive_desc_query_pd((const_mkldnn_primitive_desc_t)primitive,
  (mkldnn_query_t)what, index);
  return (long)pd;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    getFormat
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_getFormat
  (JNIEnv *env, jclass cls, jlong desc)
{
  mkldnn_memory_desc_t *jni_desc = (mkldnn_memory_desc_t*)desc;
  return (int)(jni_desc->format);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    getSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_getSize
  (JNIEnv *env, jclass cls, jlong memory_primitive_desc)
{
  const_mkldnn_primitive_desc_t mpd = (const_mkldnn_primitive_desc_t)memory_primitive_desc;
  return mkldnn_memory_primitive_desc_get_size(mpd);
}

#ifdef __cplusplus
}
#endif
