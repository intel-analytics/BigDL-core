#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescCreate(
  JNIEnv *env, jclass cls,
  long op_desc, long engine,
  long hint_forward_primitive_desc)
{
  dnnl_primitive_desc_t primitive_desc;

  CHECK_EXCEPTION(env, dnnl_primitive_desc_create(
      &primitive_desc,
      (const_dnnl_op_desc_t)op_desc,
      NULL, /* attr */
      (dnnl_engine_t)engine,
      (const_dnnl_primitive_desc_t)hint_forward_primitive_desc));

  return (long)primitive_desc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescDestroy(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  CHECK(dnnl_primitive_desc_destroy(
      (dnnl_primitive_desc_t)primitive_desc));
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveCreate(
  JNIEnv *env, jclass cls,
  long primitive_desc)
{
  dnnl_primitive_t primitive;

  CHECK(
    dnnl_primitive_create(
      &primitive,
      (const_dnnl_primitive_desc_t)primitive_desc));

  return (long)primitive;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDestroy(
  JNIEnv *env, jclass cls,
  long primitive)
{
  dnnl_primitive_destroy((dnnl_primitive_t)primitive);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ReorderPrimitiveDescCreate(
  JNIEnv *env, jclass cls, long input, long input_engine, long output, long output_engine, long attr) {

     dnnl_primitive_desc_t reorder_primitive_desc;

     CHECK(dnnl_reorder_primitive_desc_create(
		     &reorder_primitive_desc,
		     (const dnnl_memory_desc_t*)input,
		     (dnnl_engine_t)input_engine,
		     (const dnnl_memory_desc_t*)output,
		     (dnnl_engine_t)output_engine,
		     (const_dnnl_primitive_attr_t)(attr)));
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
   return dnnl_memory_desc_equal(
       (const dnnl_memory_desc_t *)lhs,
       (const dnnl_memory_desc_t *)rhs);
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

     const_dnnl_primitive_desc_t primitive_desc;
     CHECK(dnnl_primitive_get_primitive_desc((const_dnnl_primitive_t)primitive, &primitive_desc));
     return (long)primitive_desc;
  }

/** Queries primitive descriptor for primitive descriptor
  *
  * @returns NULL in case of any error
  */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescQueryMd(
JNIEnv *env, jclass cls, long primitive, int what, int index) {

  const dnnl_memory_desc_t *pd;
  pd = dnnl_primitive_desc_query_md((const_dnnl_primitive_desc_t)primitive,
		  (dnnl_query_t)what, index);
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
  dnnl_memory_desc_t *jni_desc = (dnnl_memory_desc_t*)desc;
  return (int)(jni_desc->format_kind);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    getSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_getSize
  (JNIEnv *env, jclass cls, jlong memory_primitive_desc)
{
  const dnnl_memory_desc_t *mpd = (const dnnl_memory_desc_t *)memory_primitive_desc;
  return dnnl_memory_desc_get_size(mpd);
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeUnuse
  (JNIEnv *env, jclass cls, jlong dnn_desc)
{
 free((dnnl_primitive_desc_t)dnn_desc);
 return;
}


/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    PrimitiveDescCreateV2
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PrimitiveDescCreateV2
  (JNIEnv *env, jclass cls, jlong op_desc, jlong attr, jlong engine, jlong hint_desc)
{
  dnnl_primitive_desc_t primitive_desc;

  CHECK(dnnl_primitive_desc_create(
      &primitive_desc,
      (const_dnnl_op_desc_t)op_desc,
      (const_dnnl_primitive_attr_t)attr,
      (dnnl_engine_t)engine,
      (const_dnnl_primitive_desc_t)hint_desc));

  return (long)primitive_desc;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    CreatePostOps
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_CreatePostOps
  (JNIEnv *env, jclass cls)
{
  dnnl_post_ops_t post_ops;
  CHECK(dnnl_post_ops_create(&post_ops));
  return (long)post_ops;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    DestroyPostOps
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_DestroyPostOps
  (JNIEnv *env, jclass cls, jlong post_ops)
{
  dnnl_post_ops_t ptr_post_ops = (dnnl_post_ops_t)post_ops;
  CHECK(dnnl_post_ops_destroy(ptr_post_ops));
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    PostOpsAppendEltwise
 * Signature: (JFIFF)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PostOpsAppendEltwise
  (JNIEnv *env, jclass cls, jlong post_ops, jfloat scale, jint alg, jfloat alpha, jfloat beta)
{
  dnnl_post_ops_append_eltwise(
    (dnnl_post_ops_t)post_ops,
    scale,
    (dnnl_alg_kind_t)alg,
    alpha,
    beta);
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PostOpsAppendSum
  (JNIEnv *env, jclass cls, jlong post_ops, jfloat scale)
{
  dnnl_post_ops_append_sum(
    (dnnl_post_ops_t)post_ops,
    scale);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    AttrSetPostOps
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_AttrSetPostOps
  (JNIEnv *env, jclass cls, jlong attr, jlong post_ops)
{
  dnnl_primitive_attr_set_post_ops(
    (dnnl_primitive_attr_t)attr,
    (const_dnnl_post_ops_t)post_ops);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    CreateAttr
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_CreateAttr
  (JNIEnv *env, jclass cls)
{
  dnnl_primitive_attr_t attr;
  CHECK(dnnl_primitive_attr_create(&attr));
  return (long)attr;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnn
 * Method:    DestroyAttr
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_DestroyAttr
  (JNIEnv *env, jclass cls,  long attr)
{
  dnnl_primitive_attr_t ptr_attr = (dnnl_primitive_attr_t)attr;
  CHECK(dnnl_primitive_attr_destroy(ptr_attr));
}

#define PREFIX(func) Java_com_intel_analytics_bigdl_mkl_MklDnn_##func

JNIEXPORT void JNICALL PREFIX(AttrSetIntOutputRoundMode)(JNIEnv* env,
                                                         jclass cls,
                                                         long attr,
                                                         int round_mode) {
  return ;
}

JNIEXPORT void JNICALL PREFIX(AttrSetOutputScales)(JNIEnv* env,
                                                   jclass cls,
                                                   long attr,
                                                   int count,
                                                   int mask,
                                                   jfloatArray scales) {
  /* dnnl will copy the j_scales to the internal buffer, so no need to worry
   * about the memory address moving by GC*/
  float* j_scales = (*env)->GetPrimitiveArrayCritical(env, scales, JNI_FALSE);

  CHECK_EXCEPTION(env,
                  dnnl_primitive_attr_set_output_scales(
                      (dnnl_primitive_attr_t)attr, count, mask, j_scales));
  (*env)->ReleasePrimitiveArrayCritical(env, scales, j_scales, 0);
}

#ifdef __cplusplus
}
#endif
