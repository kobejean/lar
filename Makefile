SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
	QUIET=
else
	QUIET=-s --no-print-directory
endif

CMAKE_ARGS=

all: build/Makefile
	$(MAKE) $(QUIET) -C build -j 10

debug: CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Debug
debug: build/Makefile
	$(MAKE) $(QUIET) -C build -j 10

clean: build/Makefile
	$(MAKE) $(QUIET) -C build clean -j 10

build/Makefile:
	@ echo "Running cmake to generate Makefile"; \
	mkdir build; \
	cd build; \
	cmake .. $(CMAKE_ARGS); \
	cd -
