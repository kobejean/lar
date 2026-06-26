#!/usr/bin/env bash
#
# Builds the opencv2 xcframework(s) used by lar-swift.
#
# Produces TWO static variants (both arm64-only, slimmed module set):
#   opencv2.xcframework            Release + debug info  (dev: fast + symbolicated)
#   opencv2-optimized.xcframework  Release, stripped      (distribution / App Clip size)
#
# Both are STATIC (no --dynamic) so SPM links them into the app rather than
# embedding them, and both have their iOS slices flattened to shallow bundles
# (opencv builds static frameworks as deep/versioned bundles, which iOS rejects).
#
# Usage:
#   ./script/build_opencv_framework.bash            # build both variants
#   ./script/build_opencv_framework.bash debug      # only opencv2.xcframework
#   ./script/build_opencv_framework.bash optimized  # only opencv2-optimized.xcframework
set -euo pipefail

ROOT=$(pwd)
FRAMEWORKS_PATH="$ROOT/build/frameworks"
OPENCV_BUILD="$ROOT/thirdparty/opencv/platforms/apple/build_xcframework.py"
WHICH="${1:-both}"

mkdir -p "$FRAMEWORKS_PATH"

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export AR=/usr/bin/ar
export RANLIB=/usr/bin/ranlib
export IPHONEOS_DEPLOYMENT_TARGET=12.0
export MACOSX_DEPLOYMENT_TARGET=10.15

# Slimmed module set + arm64-only, static. Shared by both variants.
COMMON_ARGS=(
    --without dnn --without gapi --without highgui --without ml
    --without objdetect --without photo --without stitching
    --without video --without videoio --without parallel
    --macos_archs=arm64
    --iphoneos_archs=arm64
    --iphonesimulator_archs=arm64
    --catalyst_archs=''
    --disable-bitcode
    --build_only_specified_archs
)

# iOS frameworks must be shallow bundles (Info.plist + binary at the root).
# opencv builds static frameworks as deep/versioned bundles, so flatten the
# iOS slices. The macOS slice keeps its valid versioned layout.
flatten_ios_slices() {
    local xc="$1"
    local slice fw
    for slice in ios-arm64 ios-arm64-simulator; do
        fw="$xc/$slice/opencv2.framework"
        [ -d "$fw/Versions" ] || continue
        echo "Flattening $slice ..."
        rm -f "$fw/Headers" "$fw/Modules" "$fw/Resources" "$fw/opencv2"
        mv "$fw/Versions/A/opencv2" "$fw/Versions/A/Headers" "$fw/Versions/A/Modules" "$fw/"
        mv "$fw/Versions/A/Resources/Info.plist" "$fw/Info.plist"
        rm -rf "$fw/Versions"
    done
}

# build_variant <output-name> [extra build_xcframework.py args...]
build_variant() {
    local out_name="$1"; shift
    echo "==> Building $out_name"
    rm -rf "$FRAMEWORKS_PATH/opencv2.xcframework"
    python3 "$OPENCV_BUILD" --out "$FRAMEWORKS_PATH" "${COMMON_ARGS[@]}" "$@"
    flatten_ios_slices "$FRAMEWORKS_PATH/opencv2.xcframework"
    rm -rf "$FRAMEWORKS_PATH/$out_name"
    mv "$FRAMEWORKS_PATH/opencv2.xcframework" "$FRAMEWORKS_PATH/$out_name"
    echo "==> Done: $FRAMEWORKS_PATH/$out_name"
}

if [ "$WHICH" = "debug" ] || [ "$WHICH" = "both" ]; then
    # Release with debug info: optimized at runtime, symbolicated.
    build_variant opencv2.xcframework --debug_info
fi

if [ "$WHICH" = "optimized" ] || [ "$WHICH" = "both" ]; then
    # Release, no debug info: smallest static lib for distribution / App Clip.
    build_variant opencv2-optimized.xcframework
fi

mkdir -p "$ROOT/lib"
for name in opencv2.xcframework opencv2-optimized.xcframework; do
    if [ -d "$FRAMEWORKS_PATH/$name" ]; then
        rm -rf "$ROOT/lib/$name"
        mv "$FRAMEWORKS_PATH/$name" "$ROOT/lib/$name"
        echo "Installed $ROOT/lib/$name"
    fi
done
