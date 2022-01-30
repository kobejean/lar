#!/usr/bin/env bash

PROJECT_NAME=g2o
FRAMEWORK_BUILD_DIR=`pwd`/build/g2o-framework
FRAMEWORK_NAME=g2o
FRAMEWORK_VERSION=A
INCLUDE_DIR=`pwd`/build/install/include/g2o
LIB_DIR=`pwd`/thirdparty/g2o/lib/Release
FRAMEWORKS_DIR=`pwd`/build/frameworks
RESOURCES_PATH=`pwd`/script/resources/apple/g2o/Resources
XCFRAMEWORK_PATH=`pwd`/lib/$FRAMEWORK_NAME.xcframework

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=Q74A2VY23K
    -DCMAKE_XCODE_ATTRIBUTE_SKIP_INSTALL=NO
    -DCMAKE_XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION=YES
    # -DCMAKE_XCODE_ATTRIBUTE_LLVM_LTO=YES
    -DCMAKE_XCODE_ATTRIBUTE_GCC_OPTIMIZATION_LEVEL=s
    -DEIGEN3_INCLUDE_DIR=$(pwd)/build/install/include/eigen3
    -DG2O_USE_VENDORED_CERES=ON
    -DG2O_USE_OPENGL=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DG2O_BUILD_APPS=OFF
    -DG2O_BUILD_EXAMPLES=OFF
    -DG2O_BUILD_SLAM2D_TYPES=OFF
    -DG2O_BUILD_SLAM2D_ADDON_TYPES=OFF
    -DG2O_BUILD_DATA_TYPES=OFF
    -DG2O_BUILD_SCLAM2D_TYPES=OFF
    -DG2O_BUILD_ICP_TYPES=OFF
    -DG2O_BUILD_SIM3_TYPES=OFF
)

mkdir -p $FRAMEWORKS_DIR
mkdir -p $FRAMEWORK_BUILD_DIR

build_archive() {
    DESTINATION="$1"
    PLATFORM_NAME="$2"
    SDK="$3"
    PRODUCTS_PATH=$FRAMEWORK_BUILD_DIR/archive/$PROJECT_NAME.$PLATFORM_NAME.xcarchive/Products
    FRAMEWORK_PATH=$FRAMEWORKS_DIR/$PLATFORM_NAME/$PROJECT_NAME.framework

    xcodebuild archive -project $FRAMEWORK_BUILD_DIR/$PROJECT_NAME.xcodeproj -scheme ALL_BUILD -destination "$DESTINATION" -configuration Release -sdk $SDK -archivePath "$FRAMEWORK_BUILD_DIR/archive/$PROJECT_NAME.$PLATFORM_NAME.xcarchive"
    
    cp $LIB_DIR/* $PRODUCTS_PATH
    rm -r $LIB_DIR
    FRAMEWORK_LIBS=`find $PRODUCTS_PATH -name *.a`

    ./script/generate_framework.bash -n$FRAMEWORK_NAME -o$FRAMEWORK_PATH -vA -r$RESOURCES_PATH -i$INCLUDE_DIR $FRAMEWORK_LIBS
}

rm -r $FRAMEWORK_BUILD_DIR
cmake -Sthirdparty/g2o -B$FRAMEWORK_BUILD_DIR -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 ${CMAKE_ARGS[@]}

build_archive "generic/platform=iOS" "iphoneos" "iphoneos"
build_archive "platform=iOS Simulator,name=iPhone 11" "iphonesimulator" "iphonesimulator"

rm -r $FRAMEWORK_BUILD_DIR
cmake -Sthirdparty/g2o -B$FRAMEWORK_BUILD_DIR -GXcode -DCMAKE_SYSTEM_NAME=Darwin "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" ${CMAKE_ARGS[@]}

build_archive "platform=macOS" "macos" "macosx"

xcodebuild -create-xcframework \
    -framework $FRAMEWORKS_DIR/iphoneos/$PROJECT_NAME.framework \
    -framework $FRAMEWORKS_DIR/iphonesimulator/$PROJECT_NAME.framework \
    -framework $FRAMEWORKS_DIR/macos/$PROJECT_NAME.framework \
    -output $XCFRAMEWORK_PATH
