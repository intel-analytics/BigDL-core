#include <omp.h>
#include "com_intel_analytics_bigdl_mkl_MklDnn.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnn_setNumThreads
(JNIEnv *env, jclass cls, jint num)
{
  int myMaxThreads = 1; // whatever
#pragma omp parallel
  {
    omp_set_num_threads(myMaxThreads);
    // omp_set_max_threads(myMaxThreads);
  }
}

#ifdef __cplusplus
}
#endif
