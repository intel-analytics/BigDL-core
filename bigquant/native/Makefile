CXX := g++-7

ifeq ($(OS), Windows_NT)
	PLATFORM := WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		PLATFORM := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		PLATFORM := MACOS
	endif
endif

# PLATFORM can CHOOSE WINDOWS, LINUX OR MACOS
ifeq ($(PLATFORM), WINDOWS)
	MAKE = mingw32-make
else
	MAKE = make
endif

MAKEFILE := Makefile.base

# PLATFORM can CHOOSE AUTO or MANUAL
LOAD_METHOD := AUTO

ifeq ($(LOAD_METHOD), MANUAL)
	MANUAL_LOAD = 1
else
	MANUAL_LOAD = 0
endif

runtime:
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) MANUAL_LOAD=$(MANUAL_LOAD) PLATFORM=$(PLATFORM) runtime

shared:
ifneq ($(PLATFORM), MACOS)
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) PLATFORM=$(PLATFORM) TARGET=SKYLAKE_SERVER shared
endif
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) PLATFORM=$(PLATFORM) TARGET=HASWELL shared
	$(MAKE) -f $(MAKEFILE) CXX=$(CXX) PLATFORM=$(PLATFORM) TARGET=MOBILE shared

all: runtime shared

test:
	$(CXX) $(CXXFLAGS) -I ./ tests/test_fc.cpp -L ./ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/ -o ./tests/test_fc.out -lCppUTest -lbigquant_rt
	$(CXX) $(CXXFLAGS) -I ./ tests/test_conv.cpp -L ./ -L /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/ -o ./tests/test_conv.out -lCppUTest -lbigquant_rt

clean:
	rm -rf *.so *.o *.a *.dll *.lib *.dylib
