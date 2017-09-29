#ifndef ARCH_ISA_SHUFFLE_H
#define ARCH_ISA_SHUFFLE_H

// VEC SHUFFLE for SP
#ifdef __AVX2__
#else  // __SSE4_2__
#endif

// VEC PERMUTE for INT
#if defined(AVX512)
#define PERMUTEX_EPI32 _mm512_permutexvar_epi32
#define PERMUTEX_EPI64 _mm512_permutexvar_epi64
#elif defined(__AVX2__)
#define PERMUTE_EPI32 _mm256_permutevar8x32_epi32
#define PERMUTE_SI128 _mm256_permute2f128_si256
#define PERMUTE_EPI64 _mm256_permute4x64_epi64
#else  // __SSE4_2__
// keep it empty
#endif

// VEC PERMUTE for PS
#if defined(AVX512)
#define PERMUTE_PS _mm512_permute_ps
#define PERMUTEX_PS _mm512_permutexvar_ps
#define PERMUTE2F128_PS_HALF _mm256_permute2f128_ps
#elif defined(__AVX2__)
#define PERMUTE_PS _mm256_permute_ps
#define PERMUTE_PS_HALF _mm_permute_ps
#define PERMUTE_PS128 _mm256_permute2f128_ps
#else  // __SSE4_2__
#define SHUFFLE_PS _mm_shuffle_ps
#endif

// VEC SHUFFLE for INT
#if defined(AVX512)
#define SHUFFLE_EPI8 _mm512_shuffle_epi8
#elif defined(__AVX2__)
#define SHUFFLE_EPI8 _mm256_shuffle_epi8
#define SHUFFLE_EPI8_HALF _mm_shuffle_epi8
#else  // __SSE4_2__
#define SHUFFLE_EPI32 _mm_shuffle_epi32
#define SHUFFLE_EPI8 _mm_shuffle_epi8
#endif

// VEC UNPACK
#if defined(AVX512)
#define UNPACKLO_PS_HALF _mm256_unpacklo_ps
#define UNPACKHI_PS_HALF _mm256_unpackhi_ps
#define UNPACKLO_PD_HALF _mm256_unpacklo_pd
#define UNPACKHI_PD_HALF _mm256_unpackhi_pd
#elif  defined(__AVX2__)
#define UNPACKLO_PS _mm256_unpacklo_ps
#define UNPACKHI_PS _mm256_unpackhi_ps
#define UNPACKLO_PD _mm256_unpacklo_pd
#define UNPACKHI_PD _mm256_unpackhi_pd
#define UNPACKLO_EPI16 _mm256_unpacklo_epi16
#define UNPACKHI_EPI16 _mm256_unpackhi_epi16
#define UNPACKLO_EPI32 _mm256_unpacklo_epi32
#define UNPACKHI_EPI32 _mm256_unpackhi_epi32
#define UNPACKLO_EPI64 _mm256_unpacklo_epi64
#define UNPACKHI_EPI64 _mm256_unpackhi_epi64
#else  // __SSE4_2__
#define UNPACKLO_PS _mm_unpacklo_ps
#define UNPACKHI_PS _mm_unpackhi_ps
#define UNPACKLO_PD _mm_unpacklo_pd
#define UNPACKHI_PD _mm_unpackhi_pd
#define UNPACKLO_EPI32 _mm_unpacklo_epi32
#define UNPACKLO_EPI64 _mm_unpacklo_epi64
#endif

// VEC SHIFT
#if defined(AVX512)
#define BSRLI_EPI128 _mm512_bsrli_epi128
#elif defined(__AVX2__)
#define SRLI_SI128_HALF _mm_srli_si128
#define SRLI_EPI64 _mm256_srli_epi64
#else
#define SRLI_SI128 _mm_srli_si128
#endif

#endif  // ISA_SHUFFLE_H
