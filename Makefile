SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
QUIET=
else
QUIET=-s --no-print-directory
endif

all: build/Makefile
	@ $(MAKE) $(QUIET) -C build

debug: build/Makefile
	@ $(MAKE) $(QUIET) -C build

clean: build/Makefile
	@ $(MAKE) $(QUIET) -C build clean

build/Makefile:
	@ echo "Running cmake to generate Makefile"; \
	mkdir build; \
	cd build; \
	cmake ../; \
	cd -