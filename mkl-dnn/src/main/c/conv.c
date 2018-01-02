#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConvForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long src_desc,
  long weights_desc,
  long bias_desc,
  long dst_desc,
  jintArray strides,
  jintArray padding_l,
  jintArray padding_r,
  int padding_kind)
{
  mkldnn_convolution_desc_t *conv_desc = malloc(sizeof(mkldnn_convolution_desc_t));

  int *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  int *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  int *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    mkldnn_convolution_forward_desc_init(
      conv_desc,
      prop_kind,
      alg_kind,
      (mkldnn_memory_desc_t *)src_desc,
      (mkldnn_memory_desc_t *)weights_desc,
      (mkldnn_memory_desc_t *)bias_desc,
      (mkldnn_memory_desc_t *)dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r,
      (mkldnn_padding_kind_t) padding_kind)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConvBackwardWeightsDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long src_desc,
  long diff_weights_desc,
  long diff_bias_desc,
  long diff_dst_desc,
  jintArray strides,
  jintArray padding_l,
  jintArray padding_r,
  int padding_kind)
{
  mkldnn_convolution_desc_t *conv_desc = malloc(sizeof(mkldnn_convolution_desc_t));

  int *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  int *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  int *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    mkldnn_convolution_backward_weights_desc_init(
      conv_desc,
      alg_kind,
      (mkldnn_memory_desc_t *)src_desc,
      (mkldnn_memory_desc_t *)diff_weights_desc,
      (mkldnn_memory_desc_t *)diff_bias_desc,
      (mkldnn_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r,
      (mkldnn_padding_kind_t) padding_kind)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_ConvBackwardDataDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_src_desc,
  long weights_desc,
  long diff_dst_desc,
  jintArray strides,
  jintArray padding_l,
  jintArray padding_r,
  int padding_kind)
{
  mkldnn_convolution_desc_t *conv_desc = malloc(sizeof(mkldnn_convolution_desc_t));

  int *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  int *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  int *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    mkldnn_convolution_backward_data_desc_init(
      conv_desc,
      alg_kind,
      (mkldnn_memory_desc_t *)diff_src_desc,
      (mkldnn_memory_desc_t *)weights_desc,
      (mkldnn_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r,
      (mkldnn_padding_kind_t) padding_kind)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

#ifdef __cplusplus
}
#endif
