SHELL = /bin/bash
VERBOSE = 0
ifeq ($(VERBOSE), 1)
 QUIET=
else
 QUIET=-s --no-print-directory
endif

# Force Apple Clang on macOS
ifeq ($(shell uname), Darwin)
CMAKE_COMPILER_ARGS = -DCMAKE_C_COMPILER=/usr/bin/clang \
                     -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
                     -DCMAKE_AR=/usr/bin/ar \
                     -DCMAKE_RANLIB=/usr/bin/ranlib \
                     -DCMAKE_OSX_ARCHITECTURES=arm64
else
CMAKE_COMPILER_ARGS =
endif

CMAKE_ARGS=

all: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release
all: configure
	$(MAKE) $(QUIET) -C build

compact: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release \
                      -DCMAKE_CXX_FLAGS="-Oz -flto=thin -ffunction-sections -fdata-sections -fvisibility=hidden -fvisibility-inlines-hidden" \
                      -DCMAKE_C_FLAGS="-Oz -flto=thin -ffunction-sections -fdata-sections -fvisibility=hidden" \
                      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-dead_strip -Wl,-S -flto=thin" \
                      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-dead_strip -Wl,-S -flto=thin"
compact: configure
	$(MAKE) $(QUIET) -C build -j 8

fast: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Release
fast: configure
	$(MAKE) $(QUIET) -C build -j 8

tests: CMAKE_ARGS = -DLAR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
tests: configure
	$(MAKE) $(QUIET) -C build -j 8

debug: CMAKE_ARGS = -DCMAKE_BUILD_TYPE=Debug
debug: configure
	$(MAKE) $(QUIET) -C build -j 8

clean: CMAKE_ARGS =
clean: configure
	$(MAKE) $(QUIET) -C build clean

artifacts: frameworks
	./script/build_artifacts.bash

frameworks:
	./script/build_frameworks.bash

configure:
	@echo "Running cmake to generate Makefile"; \
	mkdir -p build; \
	cd build && cmake .. $(CMAKE_COMPILER_ARGS) $(CMAKE_ARGS); \
	cd -

.PHONY: all compact fast tests debug clean artifacts frameworks configure