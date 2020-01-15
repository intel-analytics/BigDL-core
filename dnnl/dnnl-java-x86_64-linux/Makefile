# vim:noet
include ../../Makefile.common

# should include the build/include, because mkldnn will generate
# some header files there
MKLDNN_INC_DIR=../native/include -I ../native/build/include
MKLDNN_LIB_DIR=../native/build/src/

CFLAGS = -isystem ${JAVA_HOME}/include/linux
SUFFIX = so

all:
ifeq ($(PLATFORM), LINUX)
include ../Makefile.common
endif
