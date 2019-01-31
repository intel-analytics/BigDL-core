#include "utils.h"
#include "inc/com_intel_analytics_bigdl_mkl_Engine.h"

#define PREFIX(func) Java_com_intel_analytics_bigdl_mkl_Engine_##func

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL PREFIX(Create)(JNIEnv *env, jclass cls, int kind,
                                      int index) {
  mkldnn_engine_t engine;

  CHECK(
      mkldnn_engine_create(&engine, (mkldnn_engine_kind_t)kind, (size_t)index));

  return (long)engine;
}

JNIEXPORT void PREFIX(Destroy)(JNIEnv *env, jclass cls, long engine) {
  CHECK(mkldnn_engine_destroy((mkldnn_engine_t)engine));
}

#ifdef __cplusplus
}
#endif
