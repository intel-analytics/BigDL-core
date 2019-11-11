#include "utils.h"

#include <omp.h>
#include <time.h>

#define PREFIX(func) Java_com_intel_analytics_bigdl_mkl_Stream_##func

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL PREFIX(Create)(JNIEnv *env, jclass cls,
                                      long engine, int flag) {
  dnnl_stream_t stream;
  CHECK(dnnl_stream_create(&stream, (dnnl_engine_t)engine, flag));

  return (long)stream;
}

JNIEXPORT void JNICALL PREFIX(Destroy)(JNIEnv *env, jclass cls, long stream) {
  CHECK(dnnl_stream_destroy((dnnl_stream_t)stream));
  return;
}
#ifdef __cplusplus
}
#endif
