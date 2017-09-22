#ifndef BASE_H
#define BASE_H

#include <iostream>
#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <float.h>
#include <stdint.h>
#include <cassert>
#include <chrono>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bigquant.h"
#include "internal_api.h"
#include "arch/isa/instuction.h"
#include "arch/config.h"

#define UNROLL_NUM 4

#if defined(DEBUG)
#define INLINE_ATTRIBUTE
#define INLINE_SPECIFIER
#else
#if defined(__INTEL_COMPILER)
#define INLINE_ATTRIBUTE
#define INLINE_SPECIFIER inline
#elif defined(_MSC_VER_)
#define INLINE_ATTRIBUTE
#define INLINE_SPECIFIER __forceinline
#elif defined(__GNUC__)
#define INLINE_ATTRIBUTE __attribute__((always_inline))
#define INLINE_SPECIFIER inline
#endif
#endif

#if defined(__INTEL_COMPILER)
#define UNROLL_ATTRIBUTE
#define NOROLL_ATTRIBUTE
#elif defined(_MSC_VER_)
#define UNROLL_ATTRIBUTE
#define NOROLL_ATTRIBUTE
#elif defined(__GNUC__)
#define UNROLL_ATTRIBUTE
#define NOROLL_ATTRIBUTE __attribute__((optimize("no-unroll-loops")))
#endif

#endif
