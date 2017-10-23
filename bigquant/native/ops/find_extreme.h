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

#ifndef OPS_FIND_EXTREME_H
#define OPS_FIND_EXTREME_H

#include "../base.h"

template <typename DType>
void GenericFindMinMaxValue(const DType *p, size_t length, DType &min, DType &max) {
  min = FLT_MAX;
  max = -FLT_MAX;
  for (size_t i = 0; i < length; ++i) {
    DType value = p[i];
    min = std::min(min, value);
    max = std::max(max, value);
  }
}

#if defined(AVX512)
void SIMDFindMinMaxSPValue(const float *p, size_t length, float &min, float &max) {
  const static SIMDSITYPE PERMUTE_INDEX = SET_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
  if (length >= 2 * PS_OPERAND_WIDTH) {
    SIMDPSTYPE simd_min = SET1_PS(FLT_MAX);
    SIMDPSTYPE simd_max = SET1_PS(-FLT_MAX);
    for (size_t i = 0; i < length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i += PS_OPERAND_WIDTH) {
      SIMDPSTYPE data = LOADU_PS(p + i);
      simd_max = MAX_PS(simd_max, data);
      simd_min = MIN_PS(simd_min, data);
    }
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 2 + (3 << 2)));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 2 + (3 << 2)));
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 1));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 1));
    simd_max = PERMUTEX_PS(PERMUTE_INDEX, simd_max);
    simd_min = PERMUTEX_PS(PERMUTE_INDEX, simd_min);
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 2 + (3 << 2)));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 2 + (3 << 2)));
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 1));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 1));
    max = CVTSS_PS_QUARTER(EXTRACT_4XPS(simd_max, 0));
    min = CVTSS_PS_QUARTER(EXTRACT_4XPS(simd_min, 0));
    for (size_t i = length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  } else {
    min = FLT_MAX;
    max = -FLT_MAX;
    for (size_t i = 0; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  }
}
#elif defined(__AVX2__)
void SIMDFindMinMaxSPValue(const float *p, size_t length, float &min, float &max) {
  if (length >= 2 * PS_OPERAND_WIDTH) {
    SIMDPSTYPE simd_min = SET1_PS(FLT_MAX);
    SIMDPSTYPE simd_max = SET1_PS(-FLT_MAX);
    for (size_t i = 0; i < length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i += PS_OPERAND_WIDTH) {
      SIMDPSTYPE data = LOADU_PS(p + i);
      simd_max = MAX_PS(simd_max, data);
      simd_min = MIN_PS(simd_min, data);
    }
    simd_max = MAX_PS(simd_max, PERMUTE_PS128(simd_max, simd_max, 1));
    simd_min = MIN_PS(simd_min, PERMUTE_PS128(simd_min, simd_min, 1));
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 2 + (3 << 2)));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 2 + (3 << 2)));
    simd_max = MAX_PS(simd_max, PERMUTE_PS(simd_max, 1));
    simd_min = MIN_PS(simd_min, PERMUTE_PS(simd_min, 1));
    max = CVTSS_PS_HALF(EXTRACT_PS_HALF(simd_max, 0));
    min = CVTSS_PS_HALF(EXTRACT_PS_HALF(simd_min, 0));
    for (size_t i = length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  } else {
    min = FLT_MAX;
    max = -FLT_MAX;
    for (size_t i = 0; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  }
}
#else
void SIMDFindMinMaxSPValue(const float *p, size_t length, float &min, float &max) {
  if (length >= 2 * PS_OPERAND_WIDTH) {
    SIMDPSTYPE simd_min = SET1_PS(FLT_MAX);
    SIMDPSTYPE simd_max = SET1_PS(-FLT_MAX);
    for (size_t i = 0; i < length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i += PS_OPERAND_WIDTH) {
      SIMDPSTYPE data = LOADU_PS(p + i);
      simd_max = MAX_PS(simd_max, data);
      simd_min = MIN_PS(simd_min, data);
    }
    simd_max = MAX_PS(simd_max, SHUFFLE_PS(simd_max, simd_max, 2 + (3 << 2)));
    simd_min = MIN_PS(simd_min, SHUFFLE_PS(simd_min, simd_min, 2 + (3 << 2)));
    simd_max = MAX_PS(simd_max, SHUFFLE_PS(simd_max, simd_max, 1));
    simd_min = MIN_PS(simd_min, SHUFFLE_PS(simd_min, simd_min, 1));
    max = CVTSS_PS(simd_max);
    min = CVTSS_PS(simd_min);
    for (size_t i = length / PS_OPERAND_WIDTH * PS_OPERAND_WIDTH; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  } else {
    min = FLT_MAX;
    max = -FLT_MAX;
    for (size_t i = 0; i < length; ++i) {
      min = fminf(min, p[i]);
      max = fmaxf(max, p[i]);
    }
  }
}
#endif

template <typename DType>
void FindMinMaxValue(const DType *p, size_t length, DType &min, DType &max) {
  GenericFindMinMaxValue(p, length, min, max);
}

template <>
void FindMinMaxValue<float>(const float *p, size_t length, float &min, float &max) {
  SIMDFindMinMaxSPValue(p, length, min, max);
}

template <typename DType>
void OMPFindMinMaxValue(DType *p, size_t length, DType &min_value, DType &max_value) {
  DType min = FLT_MAX;
  DType max = -FLT_MAX;
#pragma omp parallel for reduction(min : min) reduction(max : max)
  for (size_t i = 0; i < length; ++i) {
    DType value = p[i];
    min = fminf(min, value);
    max = fmaxf(max, value);
  }
  // such assignment is because icpc openmp cannot support reference reduction
  min_value = min;
  max_value = max;
}

#endif
