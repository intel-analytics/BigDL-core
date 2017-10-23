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

#ifndef ARCH_ISA_DTYPE_H
#define ARCH_ISA_DTYPE_H

#if defined(AVX512)
#define SIMDSITYPE __m512i
#define SIMDSITYPEHALF __m256i
#define SIMDSITYPEQUARTER __m128i
#define SIMDPSTYPE __m512
#define SIMDPSTYPEHALF __m256
#define SIMDPSTYPEQUARTER __m128
#define SIMDPDTYPE __m512d
#define SIMDPDTYPEHALF __m256d
#define SIMDPDTYPEQUARTER __m128d

#elif defined(__AVX2__)
#define SIMDSITYPE __m256i
#define SIMDSITYPEHALF __m128i
#define SIMDPSTYPE __m256
#define SIMDPSTYPEHALF __m128
#define SIMDPDTYPE __m256d
#define SIMDPDTYPEHALF __m128d
#else
#define SIMDSITYPE __m128i
#define SIMDPSTYPE __m128
#define SIMDPDTYPE __m128d
#endif

#if defined(AVX512)
#define OPERAND_WIDTH 64
#define PS_OPERAND_WIDTH 16
#elif defined(__AVX2__)
#define OPERAND_WIDTH 32
#define PS_OPERAND_WIDTH 8
#else
#define OPERAND_WIDTH 16
#define PS_OPERAND_WIDTH 4
#endif

#endif
