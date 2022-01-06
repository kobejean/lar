#!/usr/bin/env bash

BUILD_ARGS=(
  --iphoneos_archs arm64
  --iphonesimulator_archs arm64,x86_64
  --macos_archs arm64,x86_64
  --build_only_specified_archs
  --without dnn
  --without gapi
  --without highgui
  --without ml
  --without objdetect
  --without photo
  --without stitching
  --without video
  --without videoio
)

python3 `pwd`/thirdparty/opencv/platforms/apple/build_xcframework.py --out `pwd`/build/frameworks ${BUILD_ARGS[@]}
