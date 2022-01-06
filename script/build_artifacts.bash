#!/usr/bin/env bash
if [[ ! -d build/frameworks ]]; then
    exit 1
fi

IFS=$'\n' read -r -d '' -a FRAMEWORK_PATHS < <( find build/frameworks -depth 1 -type d -name *.xcframework )

mkdir -p build/artifacts

for FRAMEWORK_PATH in "${FRAMEWORK_PATHS[@]}"
do
    FRAMEWORK_BASENAME="$(basename -- $FRAMEWORK_PATH)"
    zip -x "*.DS_Store" -r build/artifacts/$FRAMEWORK_BASENAME.zip $FRAMEWORK_PATH 
done

echo "ARTIFACTS BUILT SUCCESSFULLY"