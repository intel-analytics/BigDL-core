/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include <stdio.h>
#include <string.h>
#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    batchNormCreateForward
 * Signature: (JF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_batchNormCreateForward
(JNIEnv *env,
 jclass cls,
 jlong layout,
 jfloat eps)
{
  dnnError_t status        = E_UNIMPLEMENTED;
  dnnLayout_t jLayout      = (dnnLayout_t)layout;
  dnnPrimitive_t primitive = NULL;

  status = dnnBatchNormalizationCreateForward_F32(&primitive, NULL, jLayout, eps);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    batchNormCreateBackward
 * Signature: (JF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_batchNormCreateBackward
(JNIEnv *env,
 jclass cls,
 jlong layout,
 jfloat eps)
{
  dnnError_t status        = E_UNIMPLEMENTED;
  dnnLayout_t jLayout      = (dnnLayout_t)layout;
  dnnPrimitive_t primitive = NULL;

  status = dnnBatchNormalizationCreateBackwardData_F32(&primitive, NULL, jLayout, eps);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    batchNormCreateScaleShift
 * Signature: (JF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_batchNormCreateScaleShift
(JNIEnv *env,
 jclass cls,
 jlong layout,
 jfloat eps)
{
  dnnError_t status        = E_UNIMPLEMENTED;
  dnnLayout_t jLayout      = (dnnLayout_t)layout;
  dnnPrimitive_t primitive = NULL;

  status = dnnBatchNormalizationCreateBackwardScaleShift_F32(&primitive, NULL, jLayout, eps);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    setScaleShift
 * Signature: (I[FJ[FJJI)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_setScaleShift
(JNIEnv *env,
 jclass cls,
 jint affine,
 jfloatArray weight,
 jlong weightOffset,
 jfloatArray bias,
 jlong biasOffset,
 jlong storage,
 jint length)
{
  jfloat* jStorage = (jfloat*)storage;

  if (affine) {
    jfloat* jWeight = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, weight, 0));
    jfloat* jBias = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, bias, 0));

    int i = 0;
#pragma omp parallel for
    for (i = 0; i < length; i++) {
      jStorage[i] = *(jWeight + weightOffset + i);
      jStorage[i + length] = *(jBias + biasOffset + i); 
    }

    (*env)->ReleasePrimitiveArrayCritical(env, weight, jWeight, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, bias, jBias, 0);
  } else {
    int i;
    for (i = 0; i < length; i++) {
      jStorage[i] = 1.0;
      jStorage[i + length] = 0.0;
    }
  }
  return 0;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    setGradScaleShift
 * Signature: (I[FJ[FJJI)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_setGradScaleShift
(JNIEnv *env,
 jclass cls,
 jint affine,
 jfloatArray weight,
 jlong weightOffset,
 jfloatArray bias,
 jlong biasOffset,
 jlong storage,
 jint length)
{
  jfloat* jStorage = (jfloat*)storage;

  if (affine > 0) {
    jfloat* jWeight = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, weight, 0));
    jfloat* jBias = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, bias, 0));
    
    int i = 0;
#pragma omp parallel for
    for (i = 0; i < length; i++) {
      *(jWeight + weightOffset + i) = jStorage[i];
      *(jBias + biasOffset + i) = jStorage[i + length];
    }

    (*env)->ReleasePrimitiveArrayCritical(env, weight, jWeight, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, bias, jBias, 0);
  }
  return 0;
}
