/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <math.h>
#include <string.h>

#include <mkl_service.h>
#include <omp.h>
#include <sched.h>
#include <errno.h>


#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

#define TRUE  1
#define FALSE 0

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    getMklVersion
 * Signature: ()J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_getMklVersion
(JNIEnv *env, jclass cls)
{
  MKLVersion v;
  mkl_get_version(&v);
  int build = atoi(v.Build);

  return build;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    getAffinity
 * Signature: ()[B
 */
JNIEXPORT
  jbyteArray JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_getAffinity
(JNIEnv *env,
 jclass cls)
{
  cpu_set_t mask;
  const size_t size = sizeof(mask);

  int res = sched_getaffinity(0, size, &mask);
  if (res < 0) {
    return NULL;
  }

  jbyteArray ret = (*env)->NewByteArray(env, size);
  jbyte* bytes = (*env)->GetByteArrayElements(env, ret, 0);
  memcpy(bytes, &mask, size);

  (*env)->SetByteArrayRegion(env, ret, 0, size, bytes);

  return ret;
}

cpu_set_t *saved = NULL;

/**
 * @brief if setIt is true, than we bind current omp thread to relative core,
 *        otherwise, set affinity from backup.
 *
 * @param id omp thread number
 * @param setIt
 */
static void bindTo(unsigned id, int setIt)
{
  cpu_set_t *set;
  int ncpus = 256;

  set = CPU_ALLOC(ncpus);
  size_t size = CPU_ALLOC_SIZE(ncpus);
  CPU_ZERO_S(size, set);

  int ret = -1;

  if (setIt) {
    CPU_SET_S(id, size, set);
    ret = sched_setaffinity(0, size, set);
  }

  CPU_FREE(set);

  char *msg = NULL;
  switch (errno) {
  case EFAULT:
    msg = "A supplied memory address was invalid."; break;
  case EINVAL:
    msg = "The affinity bit mask mask contains no processors"; break;
  case EPERM:
    msg = "The calling thread does not have appropriate privileges."; break;
  case ESRCH:
    msg = "The thread whose ID is pid could not be found."; break;
  default:
    msg = "unknonwn reason";
  }
  
  if (ret < 0) {
    fprintf(stderr, "%d - %s\n", id, msg);
  }
}
/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    setAffinity
 * Signature: ([B)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_setAffinity
(JNIEnv *env,
 jclass cls)
{
#pragma omp parallel
  {
    unsigned id = omp_get_thread_num();
    bindTo(id, TRUE);
  }
}

JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_release
(JNIEnv *env,
 jclass cls)
{
#pragma omp parallel
  {
    unsigned id = omp_get_thread_num();
    bindTo(id, FALSE);
  }
}
