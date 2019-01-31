#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkldnn.h"

#define BT_BUF_SIZE 100

#define CHECK(f)                                                               \
  do {                                                                       \
    mkldnn_status_t s = f;                                                 \
    if (s != mkldnn_success) {                                             \
      int j, nptrs; \
      void *buffer[BT_BUF_SIZE]; \
      char **strings; \
      nptrs = backtrace(buffer, BT_BUF_SIZE); \
      printf("backtrace() returned %d addresses\n", nptrs); \
      strings = backtrace_symbols(buffer, nptrs); \
      if (strings == NULL) { \
        perror("backtrace_symbols"); \
        exit(EXIT_FAILURE); \
      } \
      for (j = 0; j < nptrs; j++) \
      printf("%s\n", strings[j]); \
      free(strings); \
      printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f,   \
             s);                                                         \
      exit(2);                                                           \
    }                                                                      \
  } while (0)

#define CHECK_TRUE(expr)                                                       \
    do {                                                                       \
        int e_ = expr;                                                         \
        if (!e_) {                                                             \
            printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr);          \
            exit(2);                                                           \
        }                                                                      \
    } while (0)


