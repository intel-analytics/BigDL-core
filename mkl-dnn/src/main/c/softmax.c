#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_SoftMaxForwardDescInit
  (JNIEnv *env, jclass cls, jint prop_kind, jlong src_desc, jint axis)
{
  mkldnn_softmax_desc_t *sm_desc = malloc(sizeof(mkldnn_softmax_desc_t));
  
  CHECK(
    mkldnn_softmax_forward_desc_init(
      sm_desc,
      (mkldnn_prop_kind_t)prop_kind,
      (mkldnn_memory_desc_t *)src_desc,
      axis));

  return (long)sm_desc;
}

// TODO free the pooling desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeSoftMaxDescInit
(JNIEnv *env, jclass cls, jlong sm_desc)
{
  free((mkldnn_softmax_desc_t *) sm_desc);
  return;
}


#ifdef __cplusplus
}
#endif
