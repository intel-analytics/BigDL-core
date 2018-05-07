#define _GNU_SOURCE

#include "com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity.h"

#include <jni.h>
#include <sched.h>

#define PREFIX(func) Java_com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity_##func

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity
 * Method:    setAffinity0
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL PREFIX(setAffinity0)(JNIEnv *env, jclass class, jintArray set)
{
  int length = (*env)->GetArrayLength(env, set);
  int *jni_set = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < length; i++)
  {
    CPU_SET(jni_set[i], &mask);
  }

  int ret = sched_setaffinity(0, sizeof(mask), &mask);

  (*env)->ReleasePrimitiveArrayCritical(env, set, jni_set, 0);
  return ret;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity
 * Method:    getAffinity0
 * Signature: ([II)I
 */
JNIEXPORT jint JNICALL
    PREFIX(getAffinity0)(JNIEnv *env, jclass class, jintArray set, jint length)
{
  cpu_set_t mask;

  int ret = sched_getaffinity(0, sizeof(mask), &mask);

  if (ret != -1)
  {
    int *jni_set = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);
    for (int i = 0; i < length; i++)
    {
      if (CPU_ISSET(i, &mask))
      {
        jni_set[i] = 1;
      }
    }
    (*env)->ReleasePrimitiveArrayCritical(env, set, jni_set, 0);
  }

  return ret;
}

#ifdef __cplusplus
}
#endif
