#ifndef ARCH_CPUID_H
#define ARCH_CPUID_H
#include <string>
#include "../common.h"

bool cpuid_support_feature(CPU_FEATURE f) {
	bool support_sse4_2;
  bool support_avx2;
  bool support_fma;
  bool support_avx512;
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 1;
    __asm__ (
      "cpuid"
      : "+a"(eax)
      , "=b"(ebx)
      , "=c"(ecx)
      , "=d"(edx)
    );
    support_sse4_2 = ecx & (1 << 20);
    support_fma = ecx & (1 << 12);
  }
  {
    uint32_t eax, ebx, ecx, edx;
    eax = 7; ecx = 0;
    __asm__ (
      "cpuid"
      : "+a"(eax)
      , "=b"(ebx)
      , "+c"(ecx)
      , "=d"(edx)
    );
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
  std::string cache_type_string;
  unsigned int cache_sets;
  unsigned int cache_coherency_line_size;
  unsigned int cache_physical_line_partitions;
  unsigned int cache_ways_of_associativity;
  size_t cache_total_size;
};

int cpuid_caches(int cache_id, struct cache_info& info) {
	uint32_t eax, ebx, ecx, edx;

	eax = 4; // get cache info
	ecx = cache_id; // cache id

	__asm__ (
		"cpuid" // call i386 cpuid instruction
		: "+a" (eax) // contains the cpuid command code, 4 for cache query
		, "=b" (ebx)
		, "+c" (ecx) // contains the cache id
		, "=d" (edx)
	); // generates output in 4 registers eax, ebx, ecx and edx

	int cache_type = eax & 0x1F;

	if (cache_type == 0) // end of valid cache identifiers
		return -1;

	std::string cache_type_string;
	switch (cache_type) {
		case 1: cache_type_string = "Data Cache"; break;
		case 2: cache_type_string = "Instruction Cache"; break;
		case 3: cache_type_string = "Unified Cache"; break;
		default: cache_type_string = "Unknown Type Cache"; break;
	}

  int cache_level = (eax >>= 5) & 0x7;
  int cache_is_self_initializing = (eax >>= 3) & 0x1; // does not need SW initialization
  int cache_is_fully_associative = (eax >>= 1) & 0x1;
  unsigned int cache_sets = ecx + 1;
  unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
  unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
  unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;
  size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

  info.cache_id = cache_id;
  info.cache_level = cache_level;
  info.cache_type_string = cache_type_string;
  info.cache_sets = cache_sets;
  info.cache_physical_line_partitions = cache_physical_line_partitions;
  info.cache_ways_of_associativity = cache_ways_of_associativity;
  info.cache_total_size = cache_total_size;
}

#endif
