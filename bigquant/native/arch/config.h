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

#ifndef ARCH_CONFIG_H
#define ARCH_CONFIG_H

typedef enum CPU_FEATURE { SSE4_2 = 0, AVX2_FMA = 1, AVX_512 = 2 } CPU_FEATURE;

#if defined(AVX512)
#define GEMM_SHUFFLE_KERNEL_M 8
#define GEMM_SHUFFLE_KERNEL_N 8
#define GEMM_SHUFFLE_KERNEL_K 8
#define CONV_SHUFFLE_KERNEL_M GEMM_SHUFFLE_KERNEL_M
#define CONV_SHUFFLE_KERNEL_N GEMM_SHUFFLE_KERNEL_N
#define CONV_SHUFFLE_KERNEL_K GEMM_SHUFFLE_KERNEL_K
#define FC_SHUFFLE_KERNEL_M GEMM_SHUFFLE_KERNEL_M
#define FC_SHUFFLE_KERNEL_N GEMM_SHUFFLE_KERNEL_N
#define FC_SHUFFLE_KERNEL_K GEMM_SHUFFLE_KERNEL_K
#elif defined(__AVX2__)
#define GEMM_SHUFFLE_KERNEL_M 4
#define GEMM_SHUFFLE_KERNEL_N 8
#define GEMM_SHUFFLE_KERNEL_K 8
#define CONV_SHUFFLE_KERNEL_M GEMM_SHUFFLE_KERNEL_M
#define CONV_SHUFFLE_KERNEL_N GEMM_SHUFFLE_KERNEL_N
#define CONV_SHUFFLE_KERNEL_K GEMM_SHUFFLE_KERNEL_N
#define FC_SHUFFLE_KERNEL_M 4
#define FC_SHUFFLE_KERNEL_N 8
#define FC_SHUFFLE_KERNEL_K 8
#else
#define GEMM_SHUFFLE_KERNEL_M 2
#define GEMM_SHUFFLE_KERNEL_N 2
#define GEMM_SHUFFLE_KERNEL_K 16
#define CONV_SHUFFLE_KERNEL_M GEMM_SHUFFLE_KERNEL_M
#define CONV_SHUFFLE_KERNEL_N GEMM_SHUFFLE_KERNEL_N
#define CONV_SHUFFLE_KERNEL_K GEMM_SHUFFLE_KERNEL_K
#define FC_SHUFFLE_KERNEL_M GEMM_SHUFFLE_KERNEL_M
#define FC_SHUFFLE_KERNEL_N GEMM_SHUFFLE_KERNEL_N
#define FC_SHUFFLE_KERNEL_K GEMM_SHUFFLE_KERNEL_K
#endif

#endif
