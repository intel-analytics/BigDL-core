#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EngineCreate(
  JNIEnv *env, jclass cls,
  int kind, int index)
{
  mkldnn_engine_t *engine = malloc(sizeof(mkldnn_engine_t));

  CHECK(mkldnn_engine_create(engine,
  (mkldnn_engine_kind_t)kind,
  (size_t)index));

  return (long)engine;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EngineDestroy(
  JNIEnv *env, jclass cls,
  long engine)
{
  mkldnn_engine_t *j_engine = (mkldnn_engine_t *)engine;
  CHECK(mkldnn_engine_destroy(*j_engine));
  free(j_engine);
}

#ifdef __cplusplus
}
#endif
