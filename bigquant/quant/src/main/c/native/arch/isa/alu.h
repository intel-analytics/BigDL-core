#ifndef ARCH_ISA_ALU_H
#define ARCH_ISA_ALU_H

// VEC ALU for SP
#if defined(AVX512)
#define MAX_PS _mm512_max_ps
#define MIN_PS _mm512_min_ps
#define ADD_PS _mm512_add_ps
#define SUB_PS _mm512_sub_ps
#elif defined(__AVX2__)
#define MAX_PS _mm256_max_ps
#define MIN_PS _mm256_min_ps
#define ADD_PS _mm256_add_ps
#define SUB_PS _mm256_sub_ps
#else // __SSE4_2__
#define MAX_PS _mm_max_ps
#define MIN_PS _mm_min_ps
#define ADD_PS _mm_add_ps
#define SUB_PS _mm_sub_ps
#endif

// VEC ALU for INT
#if defined(AVX512)
#define ADD_EPI32 _mm512_add_epi32
#define SUB_EPI32 _mm512_sub_epi32
#define MAX_EPI16 _mm512_max_epi16
#define MIN_EPI16 _mm512_min_epi16
#define ADD_EPI16 _mm512_add_epi16
#define SUB_EPI16 _mm512_sub_epi16
#define ADDS_EPI16 _mm512_adds_epi16
#define ABS_EPI16 _mm512_abs_epi16
#elif defined(__AVX2__)
#define ADD_EPI32 _mm256_add_epi32
#define ADD_EPI32_HALF _mm_add_epi32
#define SUB_EPI32 _mm256_sub_epi32
#define MAX_EPI16 _mm256_max_epi16
#define MIN_EPI16 _mm256_min_epi16
#define ADD_EPI16 _mm256_add_epi16
#define SUB_EPI16 _mm256_sub_epi16
#define ADDS_EPI16 _mm256_adds_epi16
#define ABS_EPI16 _mm256_abs_epi16
#define CMP_EPI16 _mm256_cmpgt_epi16
#define TESTZ_SI256 _mm256_testz_si256
#define TESTZ_SI _mm256_testz_si256
#else // __SSE4_2__
#define ADD_EPI32 _mm_add_epi32
#define SUB_EPI32 _mm_sub_epi32
#define MAX_EPI16 _mm_max_epi16
#define MIN_EPI16 _mm_min_epi16
#define ADD_EPI16 _mm_add_epi16
#define SUB_EPI16 _mm_sub_epi16
#define ADDS_EPI16 _mm_add_epi16
#define ABS_EPI16 _mm_abs_epi16
#define CMP_EPI16 _mm_cmpgt_epi16
#define TESTZ_SI128 _mm_testz_si128
#define TESTZ_SI TESTZ_SI128
#endif


// VEC BLEND for INT
#if defined(AVX512)
#define MASK_BLEND_EPI64 _mm512_mask_blend_epi64
#elif defined(__AVX2__)

#endif

// VEC convert
#if defined(AVX512)
#define CVTSS_PS_HALF _mm256_cvtss_f32
#define CVTSS_PS_QUARTER _mm_cvtss_f32
#define PSTOEPI32 _mm512_cvtps_epi32
#elif defined(__AVX2__)
#define CVTSS_PS_HALF _mm_cvtss_f32
#define CVTSS_PS _mm256_cvtss_f32
#define EPI16TOEPI32 _mm256_cvtepi16_epi32
#define EPI32TOPS _mm256_cvtepi32_ps
#define EPI32TOPS_HALF _mm_cvtepi32_ps
#define PSTOEPI32 _mm256_cvtps_epi32
#define PSTOEPI32_HALF _mm_cvtps_epi32
#else // __SSE4_2__
#define CVTSS_PS _mm_cvtss_f32
#define CVTEPI32_PS _mm_cvtepi32_ps
#define EPI16TOEPI32 _mm_cvtepi16_epi32
#define PSTOEPI32 _mm_cvttps_epi32
#endif

// VEC SP round
#ifdef __AVX2__
#define ROUND_PS _mm256_round_ps
#else
#define ROUND_PS _mm_round_ps
#endif

// VEC Reduce add
#if defined(AVX512)
#define MASK_REDUCEADD_EPI32 _mm512_mask_reduce_add_epi32
#elif defined(__AVX2__)

#else

#endif


#endif // ISA_ALU_H
