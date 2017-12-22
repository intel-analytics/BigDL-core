#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamCreate(
  JNIEnv *env, jclass cls,
  int stream_kind)
{
  mkldnn_stream_t *stream = malloc(sizeof(mkldnn_stream_t));
  CHECK(mkldnn_stream_create(stream,
                             (mkldnn_stream_kind_t)stream_kind));

  return (long)stream;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamSubmit(
  JNIEnv *env, jclass cls, long loc, int block, jlongArray primitives, int length)
{
  jlong * j_primitives = (*env)->GetPrimitiveArrayCritical(env, primitives, JNI_FALSE);
  mkldnn_stream_t *stream = (mkldnn_stream_t *)loc;
  mkldnn_primitive_t prim[length];
  int i = 0;
  while (i < length) {
    mkldnn_primitive_t *temp = (mkldnn_primitive_t *)j_primitives[i];
    prim[i] = *temp;
    i ++;
  }
  CHECK(mkldnn_stream_submit(*stream, block, prim, NULL)); // TODO here should not be NULL

  (*env)->ReleasePrimitiveArrayCritical(env, primitives, j_primitives, 0);
  return (long)loc;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamWait(
  JNIEnv *env, jclass cls, long loc, int block)
{
  mkldnn_stream_t *stream = (mkldnn_stream_t *)loc;
  CHECK(mkldnn_stream_wait(*stream, block, NULL));
  return (long)loc;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamDestroy(
  JNIEnv *env, jclass cls, long loc)
{
  mkldnn_stream_t *stream = (mkldnn_stream_t *)loc;
  CHECK(mkldnn_stream_destroy(*stream));
  return;
}
#ifdef __cplusplus
}
#endif
