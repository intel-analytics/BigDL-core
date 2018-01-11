#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamCreate(
  JNIEnv *env, jclass cls,
  int stream_kind)
{
  mkldnn_stream_t stream;
  CHECK(mkldnn_stream_create(&stream,
                             (mkldnn_stream_kind_t)stream_kind));

  return (long)stream;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamSubmit(
  JNIEnv *env, jclass cls, long stream, int length, jlongArray primitives)
{
  jlong * j_primitives = (*env)->GetPrimitiveArrayCritical(env, primitives, JNI_FALSE);

  // clock_t begin = clock();
  mkldnn_primitive_t prim[length];
  for (int i = 0; i < length; i++) {
    prim[i] = (mkldnn_primitive_t)(j_primitives[i]);
  }
  CHECK(mkldnn_stream_submit((mkldnn_stream_t)stream,
                             length, prim, NULL)); // TODO here should not be NULL
  // clock_t end = clock();
  // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  // printf("time costs: %lf\n", time_spent);
  // fflush(stdout);

  (*env)->ReleasePrimitiveArrayCritical(env, primitives, j_primitives, 0);
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamWait(
  JNIEnv *env, jclass cls, long stream, int block)
{
  CHECK(mkldnn_stream_wait((mkldnn_stream_t)stream, block, NULL));
  return stream;
}

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamRerun(
  JNIEnv *env, jclass cls, long stream)
{
  CHECK(mkldnn_stream_rerun((mkldnn_stream_t)stream, NULL));
  return stream;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_StreamDestroy(
  JNIEnv *env, jclass cls, long stream)
{
  CHECK(mkldnn_stream_destroy((mkldnn_stream_t)stream));
  return;
}
#ifdef __cplusplus
}
#endif
