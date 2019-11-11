#ifndef _UTILS_H
#define _UTILS_H

#include <execinfo.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include "dnnl.h"
#include "dnnl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/* we should export `throw_exception_if_failed` for macro CHECK_EXCEPTION */
void throw_exception_if_failed(JNIEnv* env,
                               dnnl_status_t status,
                               const char* file_name,
                               const char* func_name,
                               int line_num);
#ifdef __cplusplus
}
#endif

/* TODO replace this three macros to CHECK_EXCEPTION */
#define BT_BUF_SIZE 100
#define CHECK(f)                                                           \
  do {                                                                     \
    dnnl_status_t s = f;                                                   \
    if (s != dnnl_success) {                                               \
      int j, nptrs;                                                        \
      void* buffer[BT_BUF_SIZE];                                           \
      char** strings;                                                      \
      nptrs = backtrace(buffer, BT_BUF_SIZE);                              \
      printf("backtrace() returned %d addresses\n", nptrs);                \
      strings = backtrace_symbols(buffer, nptrs);                          \
      if (strings == NULL) {                                               \
        perror("backtrace_symbols");                                       \
        exit(EXIT_FAILURE);                                                \
      }                                                                    \
      for (j = 0; j < nptrs; j++)                                          \
        printf("%s\n", strings[j]);                                        \
      free(strings);                                                       \
      printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
      exit(2);                                                             \
    }                                                                      \
  } while (0)

#define CHECK_TRUE(expr)                                        \
  do {                                                          \
    int e_ = expr;                                              \
    if (!e_) {                                                  \
      printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
      exit(2);                                                  \
    }                                                           \
  } while (0)

#define CHECK_EXCEPTION(env, f)                                           \
  do {                                                                    \
    dnnl_status_t status = f;                                           \
    throw_exception_if_failed(env, status, __FILE__, __PRETTY_FUNCTION__, \
                              __LINE__);                                  \
  } while (0)

#endif
