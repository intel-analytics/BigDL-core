#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_RNNCellDescInit(
  JNIEnv *env, jclass cls,
  int kind,
  int f,
  int flags,
  float alpha,
  float clipping)
{
  mkldnn_rnn_cell_desc_t *rnn_cell_desc = malloc(sizeof(mkldnn_rnn_cell_desc_t));

  CHECK(
      mkldnn_rnn_cell_desc_init(
        rnn_cell_desc,
        kind,
        f,
        (unsigned int)flags,
        alpha,
        clipping)
      );

  return (long)rnn_cell_desc;
}

JNIEXPORT int JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_RNNCellGetGatesCount(
  JNIEnv *env, jclass cls,
  long rnn_cell_desc)
{
  return (int)mkldnn_rnn_cell_get_gates_count((mkldnn_rnn_cell_desc_t *)rnn_cell_desc);
}

JNIEXPORT int JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_RNNCellGetStatesCount(
  JNIEnv *env, jclass cls,
  long rnn_cell_desc)
{
  return (int)mkldnn_rnn_cell_get_states_count((mkldnn_rnn_cell_desc_t *)rnn_cell_desc);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_RNNForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind, long rnn_cell_desc,
  int direction, long src_layer_desc,
  long src_iter_desc, long weights_layer_desc,
  long weights_iter_desc, long bias_desc,
  long dst_layer_desc, long dst_iter_desc)
{
  mkldnn_rnn_desc_t *rnn_desc = malloc(sizeof(mkldnn_rnn_desc_t));

  CHECK(
      mkldnn_rnn_forward_desc_init(
        rnn_desc,
        prop_kind,
        (mkldnn_rnn_cell_desc_t *)rnn_cell_desc,
        (mkldnn_rnn_direction_t)direction,
        (mkldnn_memory_desc_t *)src_layer_desc,
        (mkldnn_memory_desc_t *)src_iter_desc,
        (mkldnn_memory_desc_t *)weights_layer_desc,
        (mkldnn_memory_desc_t *)weights_iter_desc,
        (mkldnn_memory_desc_t *)bias_desc,
        (mkldnn_memory_desc_t *)dst_layer_desc,
        (mkldnn_memory_desc_t *)dst_iter_desc)
      );

  return (long)rnn_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_RNNBackwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind, long rnn_cell_desc,
  int direction, long src_layer_desc,
  long src_iter_desc, long weights_layer_desc,
  long weights_iter_desc, long bias_desc,
  long dst_layer_desc, long dst_iter_desc,
  long diff_src_layer_desc, long diff_src_iter_desc,
  long diff_weights_layer_desc, long diff_weights_iter_desc,
  long diff_bias_desc, long diff_dst_layer_desc,
  long diff_dst_iter_desc)
{
  mkldnn_rnn_desc_t *rnn_desc = malloc(sizeof(mkldnn_rnn_desc_t));

  CHECK(
      mkldnn_rnn_backward_desc_init(
        rnn_desc,
        prop_kind,
        (mkldnn_rnn_cell_desc_t *)rnn_cell_desc,
        (mkldnn_rnn_direction_t)direction,
        (mkldnn_memory_desc_t *)src_layer_desc,
        (mkldnn_memory_desc_t *)src_iter_desc,
        (mkldnn_memory_desc_t *)weights_layer_desc,
        (mkldnn_memory_desc_t *)weights_iter_desc,
        (mkldnn_memory_desc_t *)bias_desc,
        (mkldnn_memory_desc_t *)dst_layer_desc,
        (mkldnn_memory_desc_t *)dst_iter_desc,
        (mkldnn_memory_desc_t *)diff_src_layer_desc,
        (mkldnn_memory_desc_t *)diff_src_iter_desc,
        (mkldnn_memory_desc_t *)diff_weights_layer_desc,
        (mkldnn_memory_desc_t *)diff_weights_iter_desc,
        (mkldnn_memory_desc_t *)diff_bias_desc,
        (mkldnn_memory_desc_t *)diff_dst_layer_desc,
        (mkldnn_memory_desc_t *)diff_dst_iter_desc)
      );

  return (long)rnn_desc;
}

// TODO free the RNN cell desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeRNNCellDescInit
(JNIEnv *env, jclass cls, jlong rnn_cell_desc)
{
  free((mkldnn_rnn_cell_desc_t *) rnn_cell_desc);
  return;
}

// TODO free the RNN desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_FreeRNNDescInit
(JNIEnv *env, jclass cls, jlong rnn_desc)
{
  free((mkldnn_rnn_desc_t *) rnn_desc);
  return;
}

#ifdef __cplusplus
}
#endif
