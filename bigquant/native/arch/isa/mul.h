#ifndef ARCH_ISA_MUL_H
#define ARCH_ISA_MUL_H

#include "alu.h"

// VEC MUL for SP
#if defined(AVX512)
#define MUL_PS _mm512_mul_ps
#define FMA_PS _mm512_fmadd_ps
#define FMA_PS_HALF _mm256_fmadd_ps
#define MUL_PS_HALF _mm256_mul_ps
#elif defined(__AVX2__)
#define MUL_PS _mm256_mul_ps
#define FMA_PS _mm256_fmadd_ps
#define MUL_PS_HALF _mm_mul_ps
#define FMA_PS_HALF _mm_fmadd_ps
#else  // __SSE4_2__
#define MUL_PS _mm_mul_ps
#define FMA_PS(x, y, z) ADD_PS(MUL_PS(x, y), z)
#endif

// VEC MUL for INT
#if defined(AVX512)
#define MADD_EPI8 _mm512_maddubs_epi16
#define MADD_EPI16 _mm512_madd_epi16
#elif defined(__AVX2__)
#define MADD_EPI8 _mm256_maddubs_epi16
#define MADD_EPI16 _mm256_madd_epi16
#else  // __SSE4_2__
#define MADD_EPI8 _mm_maddubs_epi16
#define MADD_EPI16 _mm_madd_epi16
#endif

#endif  // ISA_MUL_H
