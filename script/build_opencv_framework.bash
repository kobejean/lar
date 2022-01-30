#!/usr/bin/env bash

BUILD_ARGS=(
  --iphoneos_archs arm64
  --iphonesimulator_archs arm64,x86_64
  --macos_archs arm64,x86_64
  --catalyst_archs arm64
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

FRAMEWORKS_PATH=`pwd`/build/frameworks

mkdir -p $FRAMEWORKS_PATH

python3 `pwd`/thirdparty/opencv/platforms/apple/build_xcframework.py --out $FRAMEWORKS_PATH ${BUILD_ARGS[@]}

mv $FRAMEWORKS_PATH/opencv2.xcframework `pwd`/lib