# vim:noet
include ../../Makefile.common

CC = cl.exe
LD = link.exe

CFLAGS  = /EHsc /LD /I "${JAVA_HOME}\include" /I "${JAVA_HOME}\include\win32"
LDFLAGS = /DLL mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib 
SUFFIX  = dll

SOURCES_DIR = ..\src\main\c
OBJECTS_DIR = target\c
EXECUTABLE  = target\classes\libjmkl.$(SUFFIX)

SOURCES = $(SOURCES_DIR)\mkl.c
OBJECTS = $(patsubst $(SOURCES_DIR)\\%.c, $(OBJECTS_DIR)\\%.o, $(SOURCES))

all:
ifeq ($(PLATFORM), WINDOWS)
	@if not exist $(OBJECTS_DIR) md $(OBJECTS_DIR)
	$(CC) $(CFLAGS) -IC:$(SOURCES_DIR) /FoC:$(strip $(OBJECTS)) -c $(SOURCES)
	$(LD) $(LDFLAGS) /out:$(EXECUTABLE) $(OBJECTS)
endif
