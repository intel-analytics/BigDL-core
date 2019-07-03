#define _GNU_SOURCE

#include <jni.h>
#include <omp.h>
#include <sched.h>

#define PREFIX(func) \
  Java_com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity_##func

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class: com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity
 * Method:    setAffinity0
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL PREFIX(setAffinity0)(JNIEnv* env,
                                            jclass class,
                                            jintArray set) {
  int length = (*env)->GetArrayLength(env, set);
  int* jni_set = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < length; i++) {
    CPU_SET(jni_set[i], &mask);
  }

  int ret = sched_setaffinity(0, sizeof(mask), &mask);

  (*env)->ReleasePrimitiveArrayCritical(env, set, jni_set, 0);
  return ret;
}

/*
 * Class: com_intel_analytics_bigdl_mkl_hardware_platform_linux_LinuxAffinity
 * Method:    getAffinity0
 * Signature: ([II)I
 */
JNIEXPORT jint JNICALL PREFIX(getAffinity0)(JNIEnv* env,
                                            jclass class,
                                            jintArray set,
                                            jint length) {
  cpu_set_t mask;

  int ret = sched_getaffinity(0, sizeof(mask), &mask);

  if (ret != -1) {
    int* jni_set = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);
    for (int i = 0; i < length; i++) {
      if (CPU_ISSET(i, &mask)) {
        jni_set[i] = 1;
      }
    }
    (*env)->ReleasePrimitiveArrayCritical(env, set, jni_set, 0);
  }

  return ret;
}

JNIEXPORT jintArray JNICALL PREFIX(setOmpAffinity0)(JNIEnv* env,
                                                    jclass class,
                                                    jintArray set) {
  int available_cores_num = (*env)->GetArrayLength(env, set);
  int* available_cores = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);

  jintArray ret = (*env)->NewIntArray(env, available_cores_num);
  int j_ret[available_cores_num];

  for (int i = 0; i < available_cores_num; i++) {
    j_ret[i] = 0;
  }

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int core_id = available_cores[id % available_cores_num];

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);

    int isSucc = sched_setaffinity(0, sizeof(mask), &mask);

    j_ret[core_id] = isSucc;
  }

  (*env)->SetIntArrayRegion(env, ret, 0, available_cores_num, j_ret);
  (*env)->ReleasePrimitiveArrayCritical(env, set, available_cores, 0);

  return ret;
}

JNIEXPORT jobjectArray JNICALL PREFIX(getOmpAffinity0)(JNIEnv* env,
                                                       jclass class,
                                                       jintArray set) {
  int* available_cores = (*env)->GetPrimitiveArrayCritical(env, set, JNI_FALSE);

  int available_cores_num = (*env)->GetArrayLength(env, set);
  int omp_num_threads = omp_get_max_threads();

  // create a 2-D array, which omp_num_threads row and available_cores_num
  // column.
  jclass int_array_cls = (*env)->FindClass(env, "[I");
  jobjectArray result =
      (*env)->NewObjectArray(env, omp_num_threads, int_array_cls, NULL);
  jintArray tmp[omp_num_threads];
  for (int i = 0; i < omp_num_threads; i++) {
    tmp[i] = (*env)->NewIntArray(env, available_cores_num);
    (*env)->SetObjectArrayElement(env, result, i, tmp[i]);
  }

  int* j_tmp[omp_num_threads];
  for (int i = 0; i < omp_num_threads; i++) {
    j_tmp[i] = (*env)->GetPrimitiveArrayCritical(env, tmp[i], JNI_FALSE);
  }

#pragma omp parallel
  {
    cpu_set_t mask;
    int id = omp_get_thread_num();
    int* j_current_affinity = j_tmp[id];

    int ret = sched_getaffinity(0, sizeof(mask), &mask);

    for (int i = 0; i < available_cores_num; i++) {
      j_current_affinity[i] = 0;
    }

    if (ret != -1) {
      for (int i = 0; i < available_cores_num; i++) {
        if (CPU_ISSET(i, &mask)) {
          j_current_affinity[i] = 1;
        }
      }
    } else {
      j_current_affinity[0] = ret;
    }
  }

  for (int i = 0; i < omp_num_threads; i++) {
    (*env)->ReleasePrimitiveArrayCritical(env, tmp[i], j_tmp[i], 0);
  }

  (*env)->ReleasePrimitiveArrayCritical(env, set, available_cores, 0);

  return result;
}

#ifdef __cplusplus
}
#endif
