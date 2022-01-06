#!/usr/bin/env bash

# This script assumes everything has been built in `build` dir

MERGE_DIR=build/merge

FRAMEWORK_NAME=geoar
FRAMEWORK_VERSION=A
FRAMEWORK_PATH=`pwd`/lib/$FRAMEWORK_NAME.framework
XCFRAMEWORK_PATH=`pwd`/lib/$FRAMEWORK_NAME.xcframework

mkdir -p $MERGE_DIR
mkdir -p $MERGE_DIR/lib
mkdir -p $MERGE_DIR/include

# Merge libs

# IFS=$'\n' read -r -d '' -a lib_paths < <( find build/framework/archive/geoar.iOS.xcarchive/Products/*.a )
IFS=$'\n' read -r -d '' -a lib_paths < <( find lib/*.a )
IFS=$'\n' read -r -d '' -a lib_paths3 < <( find build/install/lib -name *.a )

libtool -static -o $MERGE_DIR/lib/geoar.a "${lib_paths[@]}" "${lib_paths3[@]}"

# Gather includes
manual_includes=( opencv4 eigen3 geoar ) # these we will add manually
inv_pat=$( ( IFS=$'\n'; echo "${manual_includes[*]}" ) )
IFS=$'\n' read -r -d '' -a include_src < <( ls build/install/include | grep -vF "$inv_pat" )
include_src=( 
  ${include_src[@]/#/build/install/include/}
  build/install/include/eigen3/Eigen
  build/install/include/opencv4/opencv2
)

# for include_dir in "${include_src[@]}"
# do
#   cp -r "${include_dir}" "$MERGE_DIR/include"
# done
cp -a include/geoar/. "$MERGE_DIR/include"

# Create Framework

# # Framework Anatomy from: https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPFrameworks/Concepts/FrameworkAnatomy.html
# ```
# MyFramework.framework/
#     Headers      -> Versions/Current/Headers
#     MyFramework  -> Versions/Current/MyFramework
#     Resources    -> Versions/Current/Resources
#     Versions/
#         A/
#             Headers/
#                 MyHeader.h
#             MyFramework
#             Resources/
#                 English.lproj/
#                     Documentation
#                     InfoPlist.strings
#                 Info.plist
#         Current  -> A
# ```

mkdir -p $FRAMEWORK_PATH
mkdir -p $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/Headers
mkdir -p $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/Resources
lipo -create $MERGE_DIR/lib/geoar.a -o $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/$FRAMEWORK_NAME

cp -a $MERGE_DIR/include/. $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/Headers
cp -a platform/apple/Resources/. $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION

ln -fs $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION $FRAMEWORK_PATH/Versions/Current
ln -fs $FRAMEWORK_PATH/Versions/Current/Headers $FRAMEWORK_PATH/Headers
ln -fs $FRAMEWORK_PATH/Versions/Current/Resources $FRAMEWORK_PATH/Resources
ln -fs $FRAMEWORK_PATH/Versions/Current/$FRAMEWORK_NAME $FRAMEWORK_PATH/$FRAMEWORK_NAME

xcodebuild -create-xcframework \
    -framework $FRAMEWORK_PATH \
    -output $XCFRAMEWORK_PATH