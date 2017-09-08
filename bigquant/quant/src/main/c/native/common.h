#ifndef COMMON_H
#define COMMON_H
#include "base.h"
#include "arch/config.h"
#include "arch/cpuid.h"
#ifdef NUMA
#include <numa.h>
#endif
#include "alloc.h"

static INLINE_SPECIFIER bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (0 <= a) && (a < b);
}


INLINE_SPECIFIER size_t GetConvOutSize(size_t in, size_t kernel, size_t stride, size_t pad, size_t dilation) {
  return (in + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

INLINE_SPECIFIER size_t GetAlignmentLength(size_t n, size_t alignment = 1) {
  return alignment * static_cast<size_t>(ceil(1.0 * n / alignment));
}

template <typename DType>
void ComputeMatrixSumPerRow(DType *dst, DType *src, size_t m, size_t n) {
  #pragma omp parallel for
  for (size_t i = 0; i < m; ++i) {
    DType sum = 0;
    for (size_t j = 0; j < n ; ++j) {
      sum += *(src + i * n + j);
    }
    dst[i] = sum;
  }
}


INLINE_SPECIFIER size_t GetSocketNum() {
#ifdef NUMA
  return numa_num_configured_nodes();
else
  return 1;
#endif
}

size_t GetBlockSize(size_t x, size_t y) {
  return x * y;
}

size_t GetBlockNum(size_t buffer_size, size_t tile_size, float ratio = 0.5) {
  return std::max(static_cast<size_t>(ratio * buffer_size / tile_size), static_cast<size_t>(1));
}

size_t GetThreadsNum() {
#ifdef _OPENMP
  size_t n;
  #pragma omp parallel
  {
    #pragma omp master
    {
      n = omp_get_num_threads();
    }
  }
  return n;
#else
 return 1;
#endif
}

size_t GetThreadsNumWrapper() {
  return GetThreadsNum();
}


// TODO(yan): still need some improvement, cannot detect cache relation, unified or private
template <size_t tile_m>
INLINE_SPECIFIER void GetBlocksInfo(size_t m, size_t k, size_t &m_in_l1, size_t &m_in_l2, size_t &m_in_l3) {

  size_t threads_num = GetThreadsNumWrapper();
  struct cache_info info;

  size_t block_size = GetBlockSize(tile_m, k);

  cpuid_caches(0, info);
  size_t l1_cache_size = info.cache_total_size;
  size_t block_num_per_L1 = GetBlockNum(l1_cache_size, block_size);

  cpuid_caches(2, info);
  size_t l2_cache_size = info.cache_total_size;
  size_t block_num_per_L2 = GetBlockNum(l2_cache_size, block_size) / block_num_per_L1 * block_num_per_L1;

  int ret = cpuid_caches(3, info);
  size_t l3_cache_size = info.cache_total_size;
#if defined(LLC_EXCLUSIVE)
  l3_cache_size /= threads_num;
#endif
  size_t block_num_per_L3 = GetBlockNum(l3_cache_size, block_size) / block_num_per_L2 * block_num_per_L2;

#if defined(LLC_EXCLUSIVE)
  if (ret < 0) {
	  m_in_l2 = std::max(std::min(block_num_per_L2 * tile_m, m / threads_num / tile_m * tile_m), tile_m);
	  m_in_l1 = std::max(std::min(block_num_per_L1 * tile_m, m_in_l2 / 2 / tile_m * tile_m), tile_m);
    m_in_l3 = m_in_l2;
  } else {
    m_in_l3 = std::max(std::min(block_num_per_L3 * tile_m, m), tile_m);
    m_in_l2 = std::max(std::min(block_num_per_L2 * tile_m, m_in_l3 / tile_m * tile_m), tile_m);
    m_in_l1 = std::max(std::min(block_num_per_L1 * tile_m, m_in_l2 / 2 / tile_m * tile_m), tile_m);
  }
#endif

#if defined(LLC_SHARED)
  m_in_l3 = std::max((ret < 0)? m : std::min(block_num_per_L3 * tile_m, m), tile_m);
  m_in_l2 = std::max(std::min(block_num_per_L2 * tile_m, m_in_l3 / threads_num / tile_m * tile_m), tile_m);
  m_in_l1 = std::max(std::min(block_num_per_L1 * tile_m, m_in_l2 / 2 / tile_m * tile_m), tile_m);
#endif

#if defined(DEBUG)
  std::cerr << "l3:" << l3_cache_size << " l2: " << l2_cache_size << " l1:" << l1_cache_size << std::endl;
  std::cerr << "m:" << m << " m_in_l3:" << m_in_l3 << " m_in_l2: " << m_in_l2 << " m_in_l1:" << m_in_l1 << std::endl;
#endif
}
#endif
