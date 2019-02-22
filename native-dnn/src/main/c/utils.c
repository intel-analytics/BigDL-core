#include <execinfo.h> /* for the backtrace api */
#include <jni.h>      /* for the JNIEnv */
#include <stdlib.h>

#include "mkldnn_types.h" /* for status of return value of mkldnn */

static jint throwException(JNIEnv* env, const char* message) {
  char* className = "java/lang/Exception";

  jclass exClass = (*env)->FindClass(env, className);
  return (*env)->ThrowNew(env, exClass, message);
}

static jint throwIllegalArgumentException(JNIEnv* env, const char* message) {
  char* className = "java/lang/IllegalArgumentException";

  jclass exClass = (*env)->FindClass(env, className);
  return (*env)->ThrowNew(env, exClass, message);
}

static jint throwOutOfMemoryError(JNIEnv* env, const char* message) {
  char* className = "java/lang/OutOfMemoryError";

  jclass exClass = (*env)->FindClass(env, className);
  return (*env)->ThrowNew(env, exClass, message);
}

static int throwUnsupportedaOperationException(JNIEnv* env,
                                               const char* message) {
  char* className = "java/lang/UnsupportedOperationException";

  jclass exClass = (*env)->FindClass(env, className);
  return (*env)->ThrowNew(env, exClass, message);
}

static int throwUnimplementedException(JNIEnv* env, const char* message) {
  char* className = "java/lang/UnimplementedException";

  jclass exClass = (*env)->FindClass(env, className);
  return (*env)->ThrowNew(env, exClass, message);
}

void throw_exception_if_failed(JNIEnv* env,
                               mkldnn_status_t status,
                               const char* file_name,
                               const char* func_name,
                               int line_num) {
  if (status == mkldnn_success) {
    return; /* right, do nothing */
  }

  const static int BT_BUF_SIZE = 100; /* buffer size for trace entries */

  void* addresses[BT_BUF_SIZE];
  size_t actual_entries = backtrace(addresses, BT_BUF_SIZE);
  fprintf(stderr, "Obtained %zd stack frames.\n", actual_entries);

  char** printable_representation =
      backtrace_symbols(addresses, actual_entries);
  for (int j = 0; j < actual_entries; j++) {
    fprintf(stderr, "[FRAME %d] %s\n", j, printable_representation[j]);
  }
  fprintf(stderr, "\n[%s:%d] error: %s returns %d\n", file_name, line_num,
          func_name, status);
  free(printable_representation); /* never forget to free the memory */

  switch (status) {
    case mkldnn_invalid_arguments:
      throwIllegalArgumentException(env, "[mkldnn] ivalid arguments");
      break;

    case mkldnn_out_of_memory:
      throwOutOfMemoryError(env, "[mkldnn] out of memory");
      break;

    case mkldnn_unimplemented:
      throwUnimplementedException(env, "[mkldnn] unimplemented");
      break;

    default:
      throwException(env, "[mkldnn] unknown type error");
      break;
  }
}
