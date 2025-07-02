SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
QUIET=
else
QUIET=-s --no-print-directory
endif

all: build/Makefile
	@ $(MAKE) $(QUIET) -C build -j 8

debug: build/Makefile
	@ $(MAKE) $(QUIET) -C build

clean: build/Makefile
	@ $(MAKE) $(QUIET) -C build clean

build/Makefile:
	@ echo "Running cmake to generate Makefile"; \
	cd build; \
	cmake ../; \
	cd -

