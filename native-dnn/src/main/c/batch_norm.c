#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_BatchNormForwardDescInit
  (JNIEnv *env, jclass cls, jint prop_kind, jlong src_desc, jfloat epsilon, jlong flags)
{
  mkldnn_batch_normalization_desc_t *bn_desc = malloc(sizeof(mkldnn_batch_normalization_desc_t));
  
  CHECK(
    mkldnn_batch_normalization_forward_desc_init(
      bn_desc,
      (mkldnn_prop_kind_t)prop_kind,
      (mkldnn_memory_desc_t *)src_desc,
      epsilon,
      flags));

  return (long)bn_desc;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_BatchNormBackwardDescInit
(JNIEnv *env,
 jclass cls,
 jint prop_kind,
 jlong diff_dst_desc,
 jlong src_desc,
 jfloat epsilon,
 jlong flags)
{
  mkldnn_batch_normalization_desc_t *bn_desc = malloc(sizeof(mkldnn_batch_normalization_desc_t));
  
  CHECK(
    mkldnn_batch_normalization_backward_desc_init(
      bn_desc,
      (mkldnn_prop_kind_t)prop_kind,
      (mkldnn_memory_desc_t *)diff_dst_desc,
      (mkldnn_memory_desc_t *)src_desc,
      epsilon,
      flags));

  return (long)bn_desc;
}

// TODO free the batchnorm desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeBatchNormDescInit
(JNIEnv *env, jclass cls, jlong bn_desc)
{
  free((mkldnn_batch_normalization_desc_t *) bn_desc);
  return;
}
#ifdef __cplusplus
}
#endif
