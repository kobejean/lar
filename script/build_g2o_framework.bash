#!/usr/bin/env bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CXXFLAGS="-std=c++14"
export AR=/usr/bin/ar
export RANLIB=/usr/bin/ranlib
export Qt5_DIR="/opt/homebrew/opt/qt@5/lib/cmake/Qt5"

PROJECT_NAME=g2o
FRAMEWORK_BUILD_DIR=`pwd`/build/g2o-framework
FRAMEWORK_NAME=g2o
INCLUDE_DIR=`pwd`/build/install/include
LIB_DIR=`pwd`/thirdparty/g2o/lib/Release
PRODUCTS_DIR=`pwd`/build/products
XCFRAMEWORK_PATH=`pwd`/lib/$FRAMEWORK_NAME.xcframework

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=QGB5NDP2FK
    -DCMAKE_XCODE_ATTRIBUTE_BUILD_LIBRARY_FOR_DISTRIBUTION=YES
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
    -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE"
)

rm -rf $PRODUCTS_DIR $XCFRAMEWORK_PATH
mkdir -p $PRODUCTS_DIR

build_static_library() {
    DESTINATION="$1"
    PLATFORM_NAME="$2"
    SDK="$3"
    
    PLATFORM_PRODUCTS_DIR=$PRODUCTS_DIR/$PLATFORM_NAME
    PLATFORM_LIB_DIR=$PLATFORM_PRODUCTS_DIR/usr/local/lib
    PLATFORM_INCLUDE_DIR=$PLATFORM_PRODUCTS_DIR/usr/local/include
    
    mkdir -p $PLATFORM_LIB_DIR
    mkdir -p $PLATFORM_INCLUDE_DIR
    
    echo "Building for $PLATFORM_NAME ($SDK)..."
    
    # Build the project
    xcodebuild build \
        -project $FRAMEWORK_BUILD_DIR/$PROJECT_NAME.xcodeproj \
        -scheme ALL_BUILD \
        -destination "$DESTINATION" \
        -configuration Release \
        -sdk $SDK
    
    if [ $? -ne 0 ]; then
        echo "Build failed for $PLATFORM_NAME"
        exit 1
    fi
    
    # Copy the pre-built libraries to products directory
    cp $LIB_DIR/* $PLATFORM_LIB_DIR/
    
    # Find all the built .a files and combine them into a single library
    FRAMEWORK_LIBS_ARRAY=($(find $PLATFORM_LIB_DIR -name "libg2o*.a"))
    
    echo "Found libraries for $PLATFORM_NAME:"
    printf '%s\n' "${FRAMEWORK_LIBS_ARRAY[@]}"
    
    # Combine all libraries into a single static library
    libtool -static -o $PLATFORM_LIB_DIR/lib$FRAMEWORK_NAME.a "${FRAMEWORK_LIBS_ARRAY[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Failed to combine libraries for $PLATFORM_NAME"
        exit 1
    fi
    
    # Remove individual libraries, keeping only the combined one
    find $PLATFORM_LIB_DIR -name "libg2o*.a" ! -name "lib$FRAMEWORK_NAME.a" -delete
    
    # Copy headers
    cp -R $INCLUDE_DIR/* $PLATFORM_INCLUDE_DIR/
    
    echo "Static library created for $PLATFORM_NAME: $PLATFORM_LIB_DIR/lib$FRAMEWORK_NAME.a"
    
    # Clean up the source lib directory for next platform
    rm -rf $LIB_DIR
}

# Build for iOS
echo "=== Building for iOS ==="
rm -rf $FRAMEWORK_BUILD_DIR
cmake -Sthirdparty/g2o -B$FRAMEWORK_BUILD_DIR -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 "-DCMAKE_OSX_ARCHITECTURES=arm64" ${CMAKE_ARGS[@]}
build_static_library "generic/platform=iOS" "iOS" "iphoneos"

# Build for iOS Simulator
echo "=== Building for iOS Simulator ==="
build_static_library "platform=iOS Simulator,name=iPhone 15" "iOS_Simulator" "iphonesimulator"

# Build for macOS
echo "=== Building for macOS ==="
rm -rf $FRAMEWORK_BUILD_DIR
cmake -Sthirdparty/g2o -B$FRAMEWORK_BUILD_DIR -GXcode -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 "-DCMAKE_OSX_ARCHITECTURES=arm64" ${CMAKE_ARGS[@]}
build_static_library "platform=macOS" "macOS" "macosx"

# Create XCFramework
echo "=== Creating XCFramework ==="
xcodebuild -create-xcframework \
    -library $PRODUCTS_DIR/iOS/usr/local/lib/lib$FRAMEWORK_NAME.a -headers $PRODUCTS_DIR/iOS/usr/local/include \
    -library $PRODUCTS_DIR/iOS_Simulator/usr/local/lib/lib$FRAMEWORK_NAME.a -headers $PRODUCTS_DIR/iOS_Simulator/usr/local/include \
    -library $PRODUCTS_DIR/macOS/usr/local/lib/lib$FRAMEWORK_NAME.a -headers $PRODUCTS_DIR/macOS/usr/local/include \
    -output $XCFRAMEWORK_PATH

if [ $? -eq 0 ]; then
    echo "✅ XCFramework created successfully at: $XCFRAMEWORK_PATH"
    
    # Show what was created
    echo ""
    echo "XCFramework contents:"
    find $XCFRAMEWORK_PATH -type f -name "*.a" -o -name "Info.plist" | sort
else
    echo "❌ XCFramework creation failed"
    exit 1
fi