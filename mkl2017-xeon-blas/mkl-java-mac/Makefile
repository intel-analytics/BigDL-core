# vim:noet
include ../../Makefile.common

SUFFIX = dylib

all:
ifeq ($(PLATFORM), MACOS)
override CC = icc
override CFLAGS = -I ${JAVA_HOME}/include/darwin -c -I ${JAVA_HOME}/include -std=c99 \
                  -fPIC -fstack-protector-all
override LDFLAGS = -Wall -ldl -liomp5 -shared -mkl=parallel -qopenmp \
                   -static-intel -no-intel-extensions  -L/Library/Developer/CommandLineTools/SDKs//MacOSX.sdk/usr/lib
include ../Makefile.common
endif
