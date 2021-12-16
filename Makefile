SHELL = /bin/bash

ifeq ($(VERBOSE), 1)
	QUIET=
else
	QUIET=-s --no-print-directory
endif

CMAKE=@ echo "Running cmake to generate Makefile"; \
	mkdir build; \
	cd build && cmake ..

all: 
	$(CMAKE)
	$(MAKE) $(QUIET) -C build

fast: 
	$(CMAKE)
	$(MAKE) $(QUIET) -C build -j 8

tests: 
	$(CMAKE) -DGEOAR_BUILD_TESTS=ON
	$(MAKE) $(QUIET) -C build -j 8

debug: 
	$(CMAKE) -DCMAKE_BUILD_TYPE=Debug
	$(MAKE) $(QUIET) -C build

clean: 
	$(CMAKE)
	$(MAKE) $(QUIET) -C build clean
