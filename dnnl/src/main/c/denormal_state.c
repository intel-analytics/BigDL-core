#include "utils.h"
#include <xmmintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_dnnl_DNNL_setFlushDenormalState(
  JNIEnv *env, jclass cls)
{
  // Denormals-are-zero (DAZ)
  _mm_setcsr( _mm_getcsr() | 0x0040 );
  // Flush-to-zero (FTZ)
  _mm_setcsr( _mm_getcsr() | 0x8000 );
}

#ifdef __cplusplus
}
#endif
