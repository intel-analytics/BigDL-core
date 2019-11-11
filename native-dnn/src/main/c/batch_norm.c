#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_BatchNormForwardDescInit
  (JNIEnv *env, jclass cls, jint prop_kind, jlong src_desc, jfloat epsilon, jlong flags)
{
  dnnl_batch_normalization_desc_t *bn_desc = malloc(sizeof(dnnl_batch_normalization_desc_t));
  
  CHECK(
    dnnl_batch_normalization_forward_desc_init(
      bn_desc,
      (dnnl_prop_kind_t)prop_kind,
      (dnnl_memory_desc_t *)src_desc,
      epsilon,
      flags));

  return (long)bn_desc;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_BatchNormBackwardDescInit
(JNIEnv *env,
 jclass cls,
 jint prop_kind,
 jlong diff_dst_desc,
 jlong src_desc,
 jfloat epsilon,
 jlong flags)
{
  dnnl_batch_normalization_desc_t *bn_desc = malloc(sizeof(dnnl_batch_normalization_desc_t));
  
  CHECK(
    dnnl_batch_normalization_backward_desc_init(
      bn_desc,
      (dnnl_prop_kind_t)prop_kind,
      (dnnl_memory_desc_t *)diff_dst_desc,
      (dnnl_memory_desc_t *)src_desc,
      epsilon,
      flags));

  return (long)bn_desc;
}

// TODO free the batchnorm desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeBatchNormDescInit
(JNIEnv *env, jclass cls, jlong bn_desc)
{
  free((dnnl_batch_normalization_desc_t *) bn_desc);
  return;
}
#ifdef __cplusplus
}
#endif
