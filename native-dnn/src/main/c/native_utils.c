#define _GNU_SOURCE

#include <jni.h>
#include <unistd.h>
#include <syscall.h>


#define PREFIX(x) Java_com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxNativeUtils_##x

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxNativeUtils
 * Method:    getTaskId0
 * Signature: ()I
 */
JNIEXPORT jint JNICALL PREFIX(getTaskId0)(JNIEnv *env, jclass cls)
{
  return syscall(__NR_gettid);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxNativeUtils
 * Method:    getPid0
 * Signature: ()I
 */
JNIEXPORT jint JNICALL PREFIX(getPid0)(JNIEnv *env, jclass cls)
{
  return getpid();
}


#ifdef __cplusplus
}
#endif
