#include "utils.h"
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT long JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EngineCreate(
  JNIEnv *env, jclass cls,
  int kind, int index)
{
  mkldnn_engine_t engine;

  CHECK(mkldnn_engine_create(&engine,
                             (mkldnn_engine_kind_t)kind,
                             (size_t)index));

  return (long)engine;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_EngineDestroy(
  JNIEnv *env, jclass cls,
  long engine)
{
  CHECK(mkldnn_engine_destroy( (mkldnn_engine_t)engine) );
}

#ifdef __cplusplus
}
#endif
