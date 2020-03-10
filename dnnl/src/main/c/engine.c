#include "utils.h"

#define PREFIX(func) Java_com_intel_analytics_bigdl_dnnl_Engine_##func

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL PREFIX(Create)(JNIEnv *env, jclass cls, int kind,
                                      int index) {
  dnnl_engine_t engine;

  CHECK(
      dnnl_engine_create(&engine, (dnnl_engine_kind_t)kind, (size_t)index));

  return (long)engine;
}

JNIEXPORT void PREFIX(Destroy)(JNIEnv *env, jclass cls, long engine) {
  CHECK(dnnl_engine_destroy((dnnl_engine_t)engine));
}

JNIEXPORT long PREFIX(Query)(JNIEnv *env, jclass cls, long primitive_desc) {
  dnnl_engine_t engine;

  CHECK(dnnl_primitive_desc_query((dnnl_primitive_desc_t) primitive_desc,
                                    dnnl_query_engine,
                                    0,
                                    &engine));

  return (long) engine;
}

#ifdef __cplusplus
}
#endif
