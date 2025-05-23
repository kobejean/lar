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
XCFRAMEWORK_PATH=$FRAMEWORKS_PATH/opencv2.xcframework
RESOURCES_PATH=`pwd`/script/resources/apple/opencv2/Resources

mkdir -p $FRAMEWORKS_PATH

python3 `pwd`/thirdparty/opencv/platforms/apple/build_xcframework.py --out $FRAMEWORKS_PATH ${BUILD_ARGS[@]}

# Copy correct Resources folder to each platform variant
copy_resources() {
    local PLATFORM_DIR="$1"
    local FRAMEWORK_PATH="$XCFRAMEWORK_PATH/$PLATFORM_DIR/opencv2.framework"
    
    if [ -d "$FRAMEWORK_PATH" ] && [ -d "$RESOURCES_PATH" ]; then
        echo "Copying Resources to $PLATFORM_DIR"
        rm -rf "$FRAMEWORK_PATH/Resources"
        cp -r "$RESOURCES_PATH" "$FRAMEWORK_PATH/Resources"
        echo "Copied Resources to $FRAMEWORK_PATH"
    else
        echo "Warning: Framework path $FRAMEWORK_PATH or Resources path $RESOURCES_PATH not found"
    fi
}

# Copy Resources to all platform variants
if [ -d "$XCFRAMEWORK_PATH" ]; then
    echo "Copying Resources folder to OpenCV XCFramework..."
    
    copy_resources "ios-arm64"
    copy_resources "ios-arm64_x86_64-simulator" 
    copy_resources "ios-arm64_x86_64-maccatalyst"
    copy_resources "macos-arm64_x86_64"
    
    echo "Resources copying completed"
else
    echo "Error: XCFramework not found at $XCFRAMEWORK_PATH"
    exit 1
fi

mv $FRAMEWORKS_PATH/opencv2.xcframework `pwd`/lib