SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
	QUIET=
else
	QUIET=-s --no-print-directory
endif

CMAKE_ARGS=

all: build/Makefile
	$(MAKE) $(QUIET) -C build

fast: build/Makefile
	$(MAKE) $(QUIET) -C build -j 8

debug: CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Debug
debug: build/Makefile
	$(MAKE) $(QUIET) -C build

clean: build/Makefile
	$(MAKE) $(QUIET) -C build clean

build/Makefile:
	@ echo "Running cmake to generate Makefile"; \
	mkdir build; \
	cd build; \
	cmake .. $(CMAKE_ARGS); \
	cd -
