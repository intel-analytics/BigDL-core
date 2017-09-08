#ifndef ARCH_ISA_MISC_H
#define ARCH_ISA_MISC_H
#include "./dtype.h"
// VEC SET for SP and INT

#if defined(AVX512)
#define ZEROS _mm512_setzero_si512
#define INIT(X) SIMDSITYPE X = ZEROS()
#define SET1_EPI8 _mm512_set1_epi8
#define SET1_EPI16 _mm512_set1_epi16
#define SET1_EPI32 _mm512_set1_epi32
#define SET1_PS _mm512_set1_ps
#define SET_EPI8 _mm512_set_epi8
#define SET_EPI8 _mm512_set_epi8
#define SET_EPI32 _mm512_set_epi32
#define SET_EPI64 _mm512_set_epi64
#define ZERO_PS _mm512_setzero_ps
#elif defined(__AVX2__)
#define ZEROS _mm256_setzero_si256
#define INIT(X) SIMDSITYPE X = ZEROS()
#define SET1_EPI16 _mm256_set1_epi16
#define SET1_EPI32 _mm256_set1_epi32
#define SET_EPI32 _mm256_set_epi32
#define SET_EPI32_HALF _mm_set_epi32
#define SET_EPI8 _mm256_set_epi8
#define SET_EPI8_HALF _mm_set_epi8
#define SET1_PS _mm256_set1_ps
#define SET1_PS_HALF _mm_set1_ps
#define ZERO_PS _mm256_setzero_ps
#else // __SSE4_2__
#define ZEROS _mm_setzero_si128
#define INIT(X) SIMDSITYPE X = ZEROS()
#define SET1_EPI16 _mm_set1_epi16
#define SET1_PS _mm_set1_ps
#define SET_EPI8 _mm_set_epi8
#define ZERO_PS _mm_setzero_ps
#endif


// Complex instructions
#ifdef __AVX2__
#define HADD_EPI32 _mm256_hadd_epi32
#else
#define HADD_EPI32 _mm_hadd_epi32
#endif

// extract

#if defined(AVX512)
#define EXTRACT_SI128 _mm512_extracti64x2_epi64
#define EXTRACT_4XPS _mm512_extractf32x4_ps
#elif defined(__AVX2__)
#define EXTRACT_EPI8 _mm256_extract_epi8
#define EXTRACT_SI128 _mm256_extracti128_si256
#define EXTRACT_EPI32 _mm256_extract_epi32
#define EXTRACT_EPI32_HALF _mm_extract_epi32
#define EXTRACT_EPI64 _mm256_extract_epi64
#define EXTRACT_PS _mm_extract_ps
#define EXTRACT_PS_HALF _mm256_extractf128_ps
#else
#define EXTRACT_EPI32 _mm_extract_epi32
#define EXTRACT_PS _mm_extract_ps
#endif






#endif // ISA_MISC_H
