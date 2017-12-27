#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EltwiseForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long data_desc,
  float alpha,
  float beta)
{
  mkldnn_eltwise_desc_t *relu_desc = malloc(sizeof(mkldnn_eltwise_desc_t));

  CHECK(
    mkldnn_eltwise_forward_desc_init(
      relu_desc,
      (mkldnn_prop_kind_t)prop_kind,
      (mkldnn_alg_kind_t)alg_kind,
      (mkldnn_memory_desc_t *)data_desc,
      alpha,
      beta));

  return (long)relu_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EltwiseBackwardDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_data_desc,
  long data_desc,
  float alpha,
  float beta)
{
  mkldnn_eltwise_desc_t *relu_desc = malloc(sizeof(mkldnn_eltwise_desc_t));

  CHECK(
    mkldnn_eltwise_backward_desc_init(
      relu_desc,
      (mkldnn_alg_kind_t)alg_kind,
      (mkldnn_memory_desc_t *)diff_data_desc,
      (mkldnn_memory_desc_t *)data_desc,
      alpha,
      beta));

  return (long)relu_desc;
}

// TODO free the eltwise desc

#ifdef __cplusplus
}
#endif
