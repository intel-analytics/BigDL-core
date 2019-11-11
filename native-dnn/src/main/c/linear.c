#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LinearForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  long data_desc,
  long weight_desc,
  long bias_desc,
  long dst_desc)
{
  dnnl_inner_product_desc_t *ip_desc = malloc(sizeof(dnnl_inner_product_desc_t));

  CHECK(
    dnnl_inner_product_forward_desc_init(
      ip_desc,
      (dnnl_prop_kind_t)prop_kind,
      (dnnl_memory_desc_t *)data_desc,
      (dnnl_memory_desc_t *)weight_desc,
      (dnnl_memory_desc_t *)bias_desc,
      (dnnl_memory_desc_t *)dst_desc));

  return (long)ip_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LinearBackwardDataDescInit(
  JNIEnv *env, jclass cls,
  long src_desc,
  long weight_desc,
  long diff_dst_desc)
{
  dnnl_inner_product_desc_t *ip_desc =
    malloc(sizeof(dnnl_inner_product_desc_t));

  CHECK(
    dnnl_inner_product_backward_data_desc_init(
      ip_desc,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)weight_desc,
      (dnnl_memory_desc_t *)diff_dst_desc));

  return (long)ip_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LinearBackwardWeightsDescInit(
  JNIEnv *env, jclass cls,
  long src_desc,
  long diff_weight_desc,
  long diff_bias_desc,
  long diff_dst_desc)
{
  dnnl_inner_product_desc_t *ip_desc =
    malloc(sizeof(dnnl_inner_product_desc_t));

  CHECK(
    dnnl_inner_product_backward_weights_desc_init(
      ip_desc,
      (dnnl_memory_desc_t *)src_desc,
      (dnnl_memory_desc_t *)diff_weight_desc,
      (dnnl_memory_desc_t *)diff_bias_desc,
      (dnnl_memory_desc_t *)diff_dst_desc));

  return (long)ip_desc;
}

// TODO free the inner product desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeLinearDescInit
(JNIEnv *env, jclass cls, jlong ip_desc)
{
  free((dnnl_inner_product_desc_t *) ip_desc);
  return;
}

#ifdef __cplusplus
}
#endif
