#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

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
  mkldnn_inner_product_desc_t *ip_desc = malloc(sizeof(mkldnn_inner_product_desc_t));

  CHECK(
    mkldnn_inner_product_forward_desc_init(
      ip_desc,
      (mkldnn_prop_kind_t)prop_kind,
      (mkldnn_memory_desc_t *)data_desc,
      (mkldnn_memory_desc_t *)weight_desc,
      (mkldnn_memory_desc_t *)bias_desc,
      (mkldnn_memory_desc_t *)dst_desc));

  return (long)ip_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LinearBackwardDataDescInit(
  JNIEnv *env, jclass cls,
  long src_desc,
  long weight_desc,
  long diff_dst_desc)
{
  mkldnn_inner_product_desc_t *ip_desc =
    malloc(sizeof(mkldnn_inner_product_desc_t));

  CHECK(
    mkldnn_inner_product_backward_data_desc_init(
      ip_desc,
      (mkldnn_memory_desc_t *)src_desc,
      (mkldnn_memory_desc_t *)weight_desc,
      (mkldnn_memory_desc_t *)diff_dst_desc));

  return (long)ip_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LinearBackwardWeightsDescInit(
  JNIEnv *env, jclass cls,
  long src_desc,
  long diff_weight_desc,
  long diff_bias_desc,
  long diff_dst_desc)
{
  mkldnn_inner_product_desc_t *ip_desc =
    malloc(sizeof(mkldnn_inner_product_desc_t));

  CHECK(
    mkldnn_inner_product_backward_weights_desc_init(
      ip_desc,
      (mkldnn_memory_desc_t *)src_desc,
      (mkldnn_memory_desc_t *)diff_weight_desc,
      (mkldnn_memory_desc_t *)diff_bias_desc,
      (mkldnn_memory_desc_t *)diff_dst_desc));

  return (long)ip_desc;
}

// TODO free the eltwise desc

#ifdef __cplusplus
}
#endif
