#ifndef ARCH_ISA_STORE_H
#define ARCH_ISA_STORE_H

#ifdef __AVX2__ // store sp
#define STOREU256_PS _mm256_storeu_ps
#define STOREU_PS STOREU256_PS
#define STORE256_PS _mm256_store_ps
#define STORE_PS STORE256_PS
#define STOREU256_PS_HALF _mm_store_ps
#define STREAMSTORE_PS _mm256_stream_ps
#define STORE_PS_HALF _mm_store_ss
#else // __SSE4_2__
#define STOREU128_PS _mm_storeu_ps
#define STOREU_PS STOREU128_PS
#define STORE128_PS _mm_store_ps
#define STORE_PS STORE128_PS
#endif


#if defined(AVX512)
#define STOREU_SI_QUARTER _mm_storeu_si128

#elif defined(__AVX2__) // store integer
#define STORE_SI256 _mm256_store_si256
#define STORE_SI STORE_SI128
#define STOREU_SI256 _mm256_storeu_si256
#define STOREU_SI STOREU_SI256
#define STORELO_EPI64_HALF _mm_storel_epi64
#else // __SSE4_2__
#define STORE_SI128 _mm_store_si128
#define STORE_SI STOREU_SI128
#define STOREU_SI128 _mm_storeu_si128
#define STOREU_SI STOREU_SI128
#define STORELO_EPI64 _mm_storel_epi64
#endif

#endif // ISA_STORE_H
