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

#ifndef ARCH_ISA_STORE_H
#define ARCH_ISA_STORE_H

#if defined(AVX512)
#define STOREU512_PS _mm512_storeu_ps
#define STOREU_PS STOREU512_PS
#define STOREU_PS_HALF _mm256_storeu_ps
#elif defined(__AVX2__)  // store sp
#define STOREU256_PS _mm256_storeu_ps
#define STOREU_PS STOREU256_PS
#define STORE256_PS _mm256_store_ps
#define STORE_PS STORE256_PS
#define STOREU256_PS_HALF _mm_storeu_ps
#define STREAMSTORE_PS _mm256_stream_ps
#define STORE_PS_HALF _mm_store_ss
#else  // __SSE4_2__
#define STOREU128_PS _mm_storeu_ps
#define STOREU_PS STOREU128_PS
#define STORE128_PS _mm_store_ps
#define STORE_PS STORE128_PS
#endif

#if defined(AVX512)
#define STOREU_SI_QUARTER _mm_storeu_si128
#define STORELO_EPI64_QUARTER _mm_storel_epi64
#elif defined(__AVX2__)  // store integer
#define STORE_SI256 _mm256_store_si256
#define STORE_SI STORE_SI128
#define STOREU_SI256 _mm256_storeu_si256
#define STOREU_SI STOREU_SI256
#define STORELO_EPI64_HALF _mm_storel_epi64
#else  // __SSE4_2__
#define STORE_SI128 _mm_store_si128
#define STORE_SI STOREU_SI128
#define STOREU_SI128 _mm_storeu_si128
#define STOREU_SI STOREU_SI128
#define STORELO_EPI64 _mm_storel_epi64
#endif

#endif  // ISA_STORE_H
