#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PoolingForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long src_desc,
  long dst_desc,
  jintArray strides,
  jintArray kernel,
  jintArray padding_l,
  jintArray padding_r,
  int padding_kind)
{
  mkldnn_pooling_desc_t *pool_desc = malloc(sizeof(mkldnn_pooling_desc_t));

  int *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  int *j_kernel = (*env)->GetPrimitiveArrayCritical(env, kernel, JNI_FALSE);
  int *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  int *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
      mkldnn_pooling_forward_desc_init(
        pool_desc,
        prop_kind,
        alg_kind,
        (mkldnn_memory_desc_t *)src_desc,
        (mkldnn_memory_desc_t *)dst_desc,
        j_strides,
        j_kernel,
        j_padding_l,
        j_padding_r,
        (mkldnn_padding_kind_t) padding_kind)
      );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, kernel, j_kernel, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)pool_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_PoolingBackwardDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_src_desc,
  long diff_dst_desc,
  jintArray strides,
  jintArray kernel,
  jintArray padding_l,
  jintArray padding_r,
  int padding_kind)
{
    mkldnn_pooling_desc_t *pool_desc = malloc(sizeof(mkldnn_pooling_desc_t));

  int *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  int *j_kernel = (*env)->GetPrimitiveArrayCritical(env, kernel, JNI_FALSE);
  int *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  int *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    mkldnn_pooling_backward_desc_init(
      pool_desc,
      alg_kind,
      (mkldnn_memory_desc_t *)diff_src_desc,
      (mkldnn_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_kernel,
      j_padding_l,
      j_padding_r,
      (mkldnn_padding_kind_t) padding_kind)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, kernel, j_kernel, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)pool_desc;
}

#ifdef __cplusplus
}
#endif
