SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
	QUIET=
else
	QUIET=-s --no-print-directory
endif

CMAKE_ARGS=

all: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release
all: configure
	$(MAKE) $(QUIET) -C build

fast: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release
fast: configure
	$(MAKE) $(QUIET) -C build -j 8

tests: CMAKE_ARGS = -DGEOAR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
tests: configure
	$(MAKE) $(QUIET) -C build -j 8

debug: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Debug
debug: configure
	$(MAKE) $(QUIET) -C build

clean: CMAKE_ARGS =
clean: configure
	$(MAKE) $(QUIET) -C build clean

configure:
	@ echo "Running cmake to generate Makefile"; \
	mkdir build; \
	cd build && cmake .. $(CMAKE_ARGS); \
	cd -
