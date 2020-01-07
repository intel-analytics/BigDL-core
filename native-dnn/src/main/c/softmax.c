#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_SoftMaxForwardDescInit
  (JNIEnv *env, jclass cls, jint prop_kind, jlong src_desc, jint axis)
{
  dnnl_softmax_desc_t *sm_desc = malloc(sizeof(dnnl_softmax_desc_t));

  CHECK(
      dnnl_softmax_forward_desc_init(
          sm_desc,
          (dnnl_prop_kind_t)prop_kind,
          (dnnl_memory_desc_t *)src_desc,
          axis));

  return (long)sm_desc;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_SoftMaxBackwardDescInit
(JNIEnv *env, jclass cls, jlong diff_desc, jlong dst_desc, jint axis)
{
  dnnl_softmax_desc_t *sm_desc = malloc(sizeof(dnnl_softmax_desc_t));

  CHECK(
      dnnl_softmax_backward_desc_init(
          sm_desc,
          (dnnl_memory_desc_t *)diff_desc,
          (dnnl_memory_desc_t *)dst_desc,
          axis));

  return (long)sm_desc;
}

// TODO free the pooling desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeSoftMaxDescInit
(JNIEnv *env, jclass cls, jlong sm_desc)
{
  free((dnnl_softmax_desc_t *) sm_desc);
  return;
}


#ifdef __cplusplus
}
#endif
