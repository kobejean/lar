#!/usr/bin/env bash
if [[ ! -d build/frameworks ]]; then
    exit 1
fi
ARTIFACTS_PATH=`pwd`/build/artifacts

mkdir -p $ARTIFACTS_PATH
cd build/frameworks

IFS=$'\n' read -r -d '' -a FRAMEWORK_PATHS < <( find *.xcframework -depth 0 -type d )

for FRAMEWORK_PATH in "${FRAMEWORK_PATHS[@]}"
do
    FRAMEWORK_BASENAME="$(basename -- $FRAMEWORK_PATH)"
    zip -x "*.DS_Store" -r $ARTIFACTS_PATH/$FRAMEWORK_BASENAME.zip $FRAMEWORK_PATH 
done

echo "ARTIFACTS BUILT SUCCESSFULLY"