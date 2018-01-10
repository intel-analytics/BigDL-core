#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Initializes an @p lrn_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training or #mkldnn_forward_inference),
 * @p alg_kind, memory descriptor @p data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k. */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LRNForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind,
  int alg_kind,
  long data_desc,
  int local_size,
  float alpha,
  float beta,
  float k)
{
  mkldnn_lrn_desc_t *lrn_desc = malloc(sizeof(mkldnn_lrn_desc_t));

  CHECK(
    mkldnn_lrn_forward_desc_init(
      lrn_desc,
      prop_kind,
      alg_kind,
      (mkldnn_memory_desc_t *)data_desc,
      local_size,
      alpha,
      beta,
      k)
    );

  return (long)lrn_desc;
}


/** Initializes an @p lrn_desc for backward propagation using @p alg_kind,
 * memory descriptors @p data_desc, and @p diff_data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k. */
JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_LRNBackwardDescInit(
  JNIEnv *env, jclass cls,
  int alg_kind,
  long diff_data_desc,
  long data_desc,
  int local_size,
  float alpha,
  float beta,
  float k)
{
  mkldnn_lrn_desc_t *lrn_desc = malloc(sizeof(mkldnn_lrn_desc_t));

  CHECK(
    mkldnn_lrn_backward_desc_init(
      lrn_desc,
      alg_kind,
      (mkldnn_memory_desc_t *)diff_data_desc,
      (mkldnn_memory_desc_t *)data_desc,
      local_size,
      alpha,
      beta,
      k)
    );

  return (long)lrn_desc;
}

#ifdef __cplusplus
}
#endif
