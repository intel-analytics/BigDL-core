# vim:noet
include ../../Makefile.common

CC = cl.exe
LD = link.exe

CFLAGS  = /EHsc /LD /I "${JAVA_HOME}\include" /I "${JAVA_HOME}\include\win32"
LDFLAGS = /LIBPATH:.\target\classes libbigquant_rt.lib /DLL
SUFFIX  = dll
MAKE    = mingw32-make
SHELL   = cmd

SOURCES_DIR = ..\src\main\c
OBJECTS_DIR = target\c
EXECUTABLE  = target\classes\libbigquant.$(SUFFIX)

SOURCES = $(SOURCES_DIR)\quant.c
OBJECTS = $(patsubst $(SOURCES_DIR)\\%.c, $(OBJECTS_DIR)\\%.o, $(SOURCES))

NATIVE_DIR = ..\native

all:
ifeq ($(PLATFORM), WINDOWS)
	cd $(NATIVE_DIR) && $(MAKE) -f Makefile LOAD_METHOD=MANUAL all
	@copy $(NATIVE_DIR)\*.$(SUFFIX) .\target\classes
	@copy $(NATIVE_DIR)\*.lib .\target\classes
	@if not exist $(OBJECTS_DIR) md $(OBJECTS_DIR)
	$(CC) $(CFLAGS) -IC:$(SOURCES_DIR) /FoC:$(strip $(OBJECTS)) -c $(SOURCES)
	$(LD) $(LDFLAGS) /out:$(EXECUTABLE) $(OBJECTS)
endif
