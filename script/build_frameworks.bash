#!/usr/bin/env bash
if [[ ! -d build ]]; then
    make fast
fi
rm -rf build/frameworks
mkdir build/frameworks

./script/build_lar_framework.bash
./script/build_g2o_framework.bash

# Uncomment if opencv needs to be rebuilt
# Otherwise just download the framework from here: https://github.com/kobejean/lar/releases/download/v0.5.0/opencv2.xcframework.zip
# ./script/build_opencv_framework.bash