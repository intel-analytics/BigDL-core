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

// TODO free the eltwise desc

#ifdef __cplusplus
}
#endif
