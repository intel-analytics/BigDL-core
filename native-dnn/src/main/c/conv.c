#include "utils.h"

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
  jlongArray strides,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    dnnl_convolution_forward_desc_init(
      conv_desc,
      prop_kind,
      alg_kind,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)weights_desc,
      (dnnl_memory_desc_t *)bias_desc,
      (dnnl_memory_desc_t *)dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_DilatedConvForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long src_desc,
  long weights_desc,
  long bias_desc,
  long dst_desc,
  jlongArray strides,
  jlongArray dilates,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_dilates = (*env)->GetPrimitiveArrayCritical(env, dilates, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK_EXCEPTION(
    env,
    dnnl_dilated_convolution_forward_desc_init(
      conv_desc,
      prop_kind,
      alg_kind,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)weights_desc,
      (dnnl_memory_desc_t *)bias_desc,
      (dnnl_memory_desc_t *)dst_desc,
      j_strides,
      j_dilates,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, dilates, j_dilates, 0);
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
  jlongArray strides,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    dnnl_convolution_backward_weights_desc_init(
      conv_desc,
      alg_kind,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)diff_weights_desc,
      (dnnl_memory_desc_t *)diff_bias_desc,
      (dnnl_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_DilatedConvBackwardWeightsDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long src_desc,
  long diff_weights_desc,
  long diff_bias_desc,
  long diff_dst_desc,
  jlongArray strides,
  jlongArray dilates,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_dilates = (*env)->GetPrimitiveArrayCritical(env, dilates, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK_EXCEPTION(
    env,
    dnnl_dilated_convolution_backward_weights_desc_init(
      conv_desc,
      alg_kind,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)diff_weights_desc,
      (dnnl_memory_desc_t *)diff_bias_desc,
      (dnnl_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_dilates,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, dilates, j_dilates, 0);
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
  jlongArray strides,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK(
    dnnl_convolution_backward_data_desc_init(
      conv_desc,
      alg_kind,
      (dnnl_memory_desc_t *)diff_src_desc,
      (dnnl_memory_desc_t *)weights_desc,
      (dnnl_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_DilatedConvBackwardDataDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_src_desc,
  long weights_desc,
  long diff_dst_desc,
  jlongArray strides,
  jlongArray dilates,
  jlongArray padding_l,
  jlongArray padding_r)
{
  dnnl_convolution_desc_t *conv_desc = malloc(sizeof(dnnl_convolution_desc_t));

  long *j_strides = (*env)->GetPrimitiveArrayCritical(env, strides, JNI_FALSE);
  long *j_dilates = (*env)->GetPrimitiveArrayCritical(env, dilates, JNI_FALSE);
  long *j_padding_l = (*env)->GetPrimitiveArrayCritical(env, padding_l, JNI_FALSE);
  long *j_padding_r = (*env)->GetPrimitiveArrayCritical(env, padding_r, JNI_FALSE);

  CHECK_EXCEPTION(
    env,
    dnnl_dilated_convolution_backward_data_desc_init(
      conv_desc,
      alg_kind,
      (dnnl_memory_desc_t *)diff_src_desc,
      (dnnl_memory_desc_t *)weights_desc,
      (dnnl_memory_desc_t *)diff_dst_desc,
      j_strides,
      j_dilates,
      j_padding_l,
      j_padding_r)
    );

  (*env)->ReleasePrimitiveArrayCritical(env, strides, j_strides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, dilates, j_dilates, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_l, j_padding_l, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, padding_r, j_padding_r, 0);

  return (long)conv_desc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeConvDescInit
(JNIEnv *env, jclass cls, jlong conv_desc)
{
  free((dnnl_convolution_desc_t *)conv_desc);
  return;
}

#ifdef __cplusplus
}
#endif
