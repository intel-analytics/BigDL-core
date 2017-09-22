#ifndef ALLOC_H
#define ALLOC_H
#include <stdlib.h>

void aligned_malloc(void** p, size_t alignment, size_t size) {
  *p = NULL;
#if defined(_MSC_VER)
  *p = _aligned_malloc(size, alignment);
  if(*p == NULL) {
    fprintf(stderr, "Failed to Allocate Memory.\n");
    exit(-1);
  }
#elif defined(__MINGW32__)
  *p = __mingw_aligned_malloc(size, alignment);
  if(*p == NULL) {
    fprintf(stderr, "Failed to Allocate Memory.\n");
    exit(-1);
  }
#else
  if (posix_memalign(p, alignment, size) != 0) {
    fprintf(stderr, "Failed to Allocate Memory.\n");
    exit(-1);
  }
#endif
}

void aligned_free(void* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#elif defined(__MINGW32__)
  __mingw_aligned_free(p);
#else
  free(p);
#endif
}

#endif
