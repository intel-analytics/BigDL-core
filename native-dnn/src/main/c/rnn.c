#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_VanillaRNNForwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind, int activation_kind,
  int direction, long src_layer_desc,
  long src_iter_desc, long weights_layer_desc,
  long weights_iter_desc, long bias_desc,
  long dst_layer_desc, long dst_iter_desc, unsigned int flags,
  float alpha, float beta)
{
  dnnl_rnn_desc_t *rnn_desc = malloc(sizeof(dnnl_rnn_desc_t));

  CHECK(
      dnnl_vanilla_rnn_forward_desc_init(
        rnn_desc,
        prop_kind,
        activation_kind,
        (dnnl_rnn_direction_t)direction,
        (dnnl_memory_desc_t *)src_layer_desc,
        (dnnl_memory_desc_t *)src_iter_desc,
        (dnnl_memory_desc_t *)weights_layer_desc,
        (dnnl_memory_desc_t *)weights_iter_desc,
        (dnnl_memory_desc_t *)bias_desc,
        (dnnl_memory_desc_t *)dst_layer_desc,
        (dnnl_memory_desc_t *)dst_iter_desc,
        flags, alpha, beta)
      );

  return (long)rnn_desc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_VanillaRNNBackwardDescInit(
  JNIEnv *env, jclass cls,
  int prop_kind, int activation_kind,
  int direction, long src_layer_desc,
  long src_iter_desc, long weights_layer_desc,
  long weights_iter_desc, long bias_desc,
  long dst_layer_desc, long dst_iter_desc,
  long diff_src_layer_desc, long diff_src_iter_desc,
  long diff_weights_layer_desc, long diff_weights_iter_desc,
  long diff_bias_desc, long diff_dst_layer_desc,
  long diff_dst_iter_desc, unsigned flags, float alpha, float beta)
{
  dnnl_rnn_desc_t *rnn_desc = malloc(sizeof(dnnl_rnn_desc_t));

  CHECK(
      dnnl_vanilla_rnn_backward_desc_init(
        rnn_desc,
        prop_kind,
        activation_kind,
        (dnnl_rnn_direction_t)direction,
        (dnnl_memory_desc_t *)src_layer_desc,
        (dnnl_memory_desc_t *)src_iter_desc,
        (dnnl_memory_desc_t *)weights_layer_desc,
        (dnnl_memory_desc_t *)weights_iter_desc,
        (dnnl_memory_desc_t *)bias_desc,
        (dnnl_memory_desc_t *)dst_layer_desc,
        (dnnl_memory_desc_t *)dst_iter_desc,
        (dnnl_memory_desc_t *)diff_src_layer_desc,
        (dnnl_memory_desc_t *)diff_src_iter_desc,
        (dnnl_memory_desc_t *)diff_weights_layer_desc,
        (dnnl_memory_desc_t *)diff_weights_iter_desc,
        (dnnl_memory_desc_t *)diff_bias_desc,
        (dnnl_memory_desc_t *)diff_dst_layer_desc,
        (dnnl_memory_desc_t *)diff_dst_iter_desc, flags, alpha, beta)
      );

  return (long)rnn_desc;
}

// TODO free the RNN desc
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_FreeRNNDescInit
(JNIEnv *env, jclass cls, jlong rnn_desc)
{
  free((dnnl_rnn_desc_t *) rnn_desc);
  return;
}

#ifdef __cplusplus
}
#endif
