# vim:noet
include ../../Makefile.common

CFLAGS = -I ${JAVA_HOME}/include/linux
SUFFIX = so
OS     = "RedHatEnterpriseServer"
VER    = 5.4

# control whether to build
BUILD := NO

ifeq ($(PLATFORM), LINUX)

# check the os version
# because of the default shell in make is sh
# which has no [[ command, we should set it to bash first
SHELL := /bin/bash
OS_RELEASE = $(shell lsb_release -si)-$(shell lsb_release -sr)
REDHAT5 = $(shell if [[ $(OS_RELEASE) == $(OS)-$(VER) ]]; then echo 1; else echo 0; fi)

ifeq ($(REDHAT5), 1)
BUILD := YES
endif
endif

all:
ifeq ($(BUILD), YES)
include ../Makefile.common
endif
