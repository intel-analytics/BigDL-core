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

#ifndef ARCH_ISA_LOAD_H
#define ARCH_ISA_LOAD_H

#if defined(AVX512)
#define LOAD_SI512 _mm512_load_si512
#define LOAD_SI LOAD_SI512
#elif defined(__AVX2__)  // load integer_
#define LOAD_SI256 _mm256_load_si256
#define LOAD_SI LOAD_SI256
#define STREAMLOAD_SI256 _mm256_stream_load_si256
#define STREAMLOAD_SI STREAMLOAD_SI256
#else  // __SSE4_2__
#define LOAD_SI128 _mm_load_si128
#define LOAD_SI LOAD_SI128
#define STREAMLOAD_SI128 _mm_stream_load_si128
#define STREAMLOAD_SI STREAMLOAD_SI128
#define LOADU_SI128 _mm_loadu_si128
#endif

#if defined(AVX512)
#define LOADU512_PS _mm512_loadu_ps
#define LOADU_PS LOADU512_PS
#define LOADU_PS_HALF _mm256_loadu_ps

#elif defined(__AVX2__)  // load ps
#define LOAD256_PS _mm256_load_ps
#define LOAD_PS LOAD256_PS
#define LOADU256_PS _mm256_loadu_ps
#define LOADU_PS LOADU256_PS
#define LOADU_PS_HALF _mm_loadu_ps
#define STREAMLOAD256_PS _mm256_load_ps
#define STREAMLOAD_PS STREAMLOAD256_PS
#else  // __SSE4_2__
#define LOADU128_PS _mm_loadu_ps
#define LOADU_PS LOADU128_PS
#endif

// Broadcast
#if defined(AVX512)

#elif defined(__AVX2__)
#define BROADCASTLOAD_PD _mm256_broadcast_sd
#else

#endif

#endif  // ISA_LOAD_H
