#ifndef ARCH_CPUID_H
#define ARCH_CPUID_H
#include <string>
#include "../common.h"

static bool cpuid_support_feature(CPU_FEATURE f) {
  bool support_sse4_2;
  bool support_avx2;
  bool support_fma;
  bool support_avx512;
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 1;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx));
    support_sse4_2 = ecx & (1 << 20);
    support_fma = ecx & (1 << 12);
  }
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 7;
    ecx = 0;
    __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
    support_avx2 = ebx & (1 << 5);
    support_avx512 = ebx & ((1 << 16) + (1 << 17) + (1 << 30) + (1 << 31));
  }
  if (f == SSE4_2) {
    return support_sse4_2;
  } else if (f == AVX2_FMA) {
    return support_fma && support_avx2;
  } else if (f == AVX_512) {
    return support_avx512;
  } else {
    throw "Unknown CPU ISA. Internal Error.\n";
  }
}

struct cache_info {
  int cache_id;
  int cache_level;
  size_t cache_size;
  size_t logic_cores_per_package;
  bool hyper_threading;
  bool inclusive;
};

static int cpuid_caches(int cache_id, struct cache_info& info) {
  uint32_t eax, ebx, ecx, edx;

  eax = 4;         // get cache info
  ecx = cache_id;  // cache id

  __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));

  int cache_type = eax & 31;

  if (cache_type == 0) return -1;

  size_t cache_level = (eax >> 5) & 0x7;
  size_t cache_sets = ecx + 1;
  size_t cacheline_size = (ebx & 0xfff) + 1;
  size_t cacheline_partitions = ((ebx >> 12) & 0x3ff) + 1;
  size_t cache_ways = ((ebx >> 22) & 0x3ff) + 1;
  size_t cache_size = cache_ways * cacheline_partitions * cacheline_size * cache_sets;
  bool inclusive = (edx >> 1) & 1;

  info.cache_id = cache_id;
  info.cache_level = cache_level;
  info.cache_size = cache_size;
  info.inclusive = inclusive;

  eax = 0xb;
  ecx = 1;
  __asm__("cpuid" : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx));
  info.logic_cores_per_package = ebx;
}

#endif
