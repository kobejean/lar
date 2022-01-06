#!/usr/bin/env bash
if [[ ! -d build ]]; then
    make fast
fi
mkdir build/frameworks

./script/build_g2o_framework.bash
./script/build_geoar_framework.bash

# Uncomment if opencv needs to be rebuilt
# Otherwise just download the framework from here:
# ./script/build_opencv_framework.bash