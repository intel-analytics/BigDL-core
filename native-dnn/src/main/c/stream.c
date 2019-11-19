#include "utils.h"

#include <omp.h>
#include <time.h>

#define PREFIX(func) Java_com_intel_analytics_bigdl_dnnl_Stream_##func

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

JNIEXPORT void JNICALL PREFIX(Submit)(JNIEnv *env, jclass cls, long primitive, long stream,
                                      int length, jintArray indexes, jlongArray memorys) {
  jlong *j_memorys =
      (*env)->GetPrimitiveArrayCritical(env, memorys, JNI_FALSE);
  jint *j_indexes =
    (*env)->GetPrimitiveArrayCritical(env, indexes, JNI_FALSE);

  dnnl_exec_arg_t args[length];

  for (int i = 0; i < length; i++) {
    args[i].memory = (dnnl_memory_t)(j_memorys[i]);
    args[i].arg = j_indexes[i];
  }

  CHECK_EXCEPTION(env,
                  dnnl_primitive_execute(
                    (dnnl_primitive_t)primitive,
                    (dnnl_stream_t)stream,
                    length, &args[0]));

  (*env)->ReleasePrimitiveArrayCritical(env, memorys, j_memorys, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, indexes, j_indexes, 0);
}

#ifdef __cplusplus
}
#endif
