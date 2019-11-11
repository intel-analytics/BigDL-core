#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_EltwiseForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long data_desc,
  float alpha,
  float beta)
{
  dnnl_eltwise_desc_t *relu_desc = malloc(sizeof(dnnl_eltwise_desc_t));

  CHECK(
    dnnl_eltwise_forward_desc_init(
      relu_desc,
      (dnnl_prop_kind_t)prop_kind,
      (dnnl_alg_kind_t)alg_kind,
      (dnnl_memory_desc_t *)data_desc,
      alpha,
      beta));

  return (long)relu_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_EltwiseBackwardDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_data_desc,
  long data_desc,
  float alpha,
  float beta)
{
  dnnl_eltwise_desc_t *relu_desc = malloc(sizeof(dnnl_eltwise_desc_t));

  CHECK(
    dnnl_eltwise_backward_desc_init(
      relu_desc,
      (dnnl_alg_kind_t)alg_kind,
      (dnnl_memory_desc_t *)diff_data_desc,
      (dnnl_memory_desc_t *)data_desc,
      alpha,
      beta));

  return (long)relu_desc;
}

// TODO free the eltwise desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeEltwiseDescInit
(JNIEnv *env, jclass cls, jlong relu_desc)
{
  free((dnnl_eltwise_desc_t *) relu_desc);
  return;
}

#ifdef __cplusplus
}
#endif
