#!/usr/bin/make -f
CXX := g++
MAKE := make
MAKEFILE := Makefile.base

runtime:
	$(MAKE) -f $(MAKEFILE) runtime

shared:
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) TARGET=HASWELL shared
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) TARGET=MOBILE shared
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) TARGET=SKYLAKE_SERVER shared

.PHONY: clean

all: runtime shared

test:
	$(CXX) $(CFLAGS) -I ./ tests/test_fc.cpp -L ./ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/ -o ./tests/test_fc.out -lCppUTest -lnnfixpoint_rt
	$(CXX) $(CFLAGS) -I ./ tests/test_conv.cpp -L ./ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/ -o ./tests/test_conv.out -lCppUTest -lnnfixpoint_rt

clean:
	rm -rf *.so *.o *.a *.dll
