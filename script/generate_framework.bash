#!/usr/bin/env bash
while getopts "n:o:v:r:i:" flag; do
  case "$flag" in
      n) FRAMEWORK_NAME=$OPTARG;;
      o) FRAMEWORK_PATH=$OPTARG;;
      v) FRAMEWORK_VERSION=$OPTARG;;
      r) RESOURCES_PATH=$OPTARG;;
      i) INCLUDE_DIR=$OPTARG;;
  esac
done

FRAMEWORK_LIBS=("${@:$OPTIND}")

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

libtool -static -o $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/$FRAMEWORK_NAME "${FRAMEWORK_LIBS[@]}"
cp -a $INCLUDE_DIR/. $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/Headers
cp -a $RESOURCES_PATH/. $FRAMEWORK_PATH/Versions/$FRAMEWORK_VERSION/Resources

ln -fs ./$FRAMEWORK_VERSION $FRAMEWORK_PATH/Versions/Current
ln -fs ./Versions/Current/Headers $FRAMEWORK_PATH/Headers
ln -fs ./Versions/Current/Resources $FRAMEWORK_PATH/Resources
ln -fs ./Versions/Current/$FRAMEWORK_NAME $FRAMEWORK_PATH/$FRAMEWORK_NAME