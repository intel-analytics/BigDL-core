#include "inc/com_intel_analytics_bigdl_mkl_Stream.h"
#include "utils.h"

#include <omp.h>
#include <time.h>

#define PREFIX(func) Java_com_intel_analytics_bigdl_mkl_Stream_##func

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL PREFIX(Create)(JNIEnv *env, jclass cls,
                                      int stream_kind) {
  mkldnn_stream_t stream;
  CHECK(mkldnn_stream_create(&stream, (mkldnn_stream_kind_t)stream_kind));

  return (long)stream;
}

JNIEXPORT void JNICALL PREFIX(Submit)(JNIEnv *env, jclass cls, long stream,
                                      int length, jlongArray primitives) {
  jlong *j_primitives =
      (*env)->GetPrimitiveArrayCritical(env, primitives, JNI_FALSE);

  mkldnn_primitive_t prim[length];
  for (int i = 0; i < length; i++) {
    prim[i] = (mkldnn_primitive_t)(j_primitives[i]);
  }

  CHECK_EXCEPTION(env,
                  mkldnn_stream_submit((mkldnn_stream_t)stream, length, prim,
                                       NULL));  // TODO here should not be NULL
  (*env)->ReleasePrimitiveArrayCritical(env, primitives, j_primitives, 0);
}

JNIEXPORT long JNICALL PREFIX(Wait)(JNIEnv *env, jclass cls, long stream,
                                    int block) {
  CHECK(mkldnn_stream_wait((mkldnn_stream_t)stream, block, NULL));
  return stream;
}

JNIEXPORT long JNICALL PREFIX(Rerun)(JNIEnv *env, jclass cls, long stream) {
  CHECK(mkldnn_stream_rerun((mkldnn_stream_t)stream, NULL));
  return stream;
}

JNIEXPORT void JNICALL PREFIX(Destroy)(JNIEnv *env, jclass cls, long stream) {
  CHECK(mkldnn_stream_destroy((mkldnn_stream_t)stream));
  return;
}
#ifdef __cplusplus
}
#endif
