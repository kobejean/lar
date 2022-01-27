#!/usr/bin/env bash
if [[ ! -d build ]]; then
    make fast
fi
mkdir build/frameworks

./script/build_geoar_framework.bash
./script/build_g2o_framework.bash

# Uncomment if opencv needs to be rebuilt
# Otherwise just download the framework from here: https://github.com/kobejean/GeoARCore/releases/download/v1.0.0-alpha.0/opencv2.xcframework.zip
# ./script/build_opencv_framework.bash