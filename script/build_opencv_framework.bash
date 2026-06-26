#!/usr/bin/env bash
#
# Builds the opencv2 xcframework(s) used by lar-swift, for OpenCV 5.
#
# Default output: STATIC-LIBRARY xcframeworks (libopencv2.a + Headers/opencv2). SPM
# *links* static-library xcframeworks and never embeds them — which is required for
# the iOS app to install. (Embedding a static .framework makes Xcode write a broken
# stub binary and the device install fails: "parse_macho_iterate_slices failed".)
# The Swift layer is opencv-free, so opencv needs no framework/Swift module — a plain
# static library + C++ headers is enough.
#
#   opencv2.xcframework            Release + debug info  (dev: fast + symbolicated)
#   opencv2-optimized.xcframework  Release, no debug info (distribution / App-Clip size)
#
# Set OPENCV_DYNAMIC=1 to instead emit embeddable *dynamic* frameworks (larger; loses
# the static/App-Clip size benefit, but a standard build with no .a repackaging).
#
# Toolchain (OpenCV 5 + Xcode 26): requires cmake < 4 and python 3.12 (see
# ensure_toolchain). OpenCV's source workarounds are applied by
# script/patches/apply_opencv5_patches.sh.
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
DYNAMIC="${OPENCV_DYNAMIC:-0}"

export CC=/usr/bin/clang CXX=/usr/bin/clang++ AR=/usr/bin/ar RANLIB=/usr/bin/ranlib
export IPHONEOS_DEPLOYMENT_TARGET=12.0
export MACOSX_DEPLOYMENT_TARGET=10.15

# OpenCV 5's -GXcode build breaks on cmake 4.x (compiler detection) and on python 3.14
# (Obj-C binding generator). Prefer uv-managed cmake 3.x + python 3.12 if present, then
# verify. Install without admin: `uv tool install "cmake<4"` and `uv python install 3.12`.
ensure_toolchain() {
    [ -x "$HOME/.local/bin/cmake" ] && export PATH="$HOME/.local/bin:$PATH"
    local py312; py312="$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/bin 2>/dev/null | head -1 || true)"
    [ -n "$py312" ] && export PATH="$py312:$PATH"

    local cmajor pyminor
    cmajor="$(cmake --version 2>/dev/null | sed -n '1s/.*version \([0-9][0-9]*\).*/\1/p')"
    pyminor="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 99)"
    if [ -z "${cmajor:-}" ] || [ "$cmajor" -ge 4 ]; then
        echo "ERROR: OpenCV's -GXcode build needs cmake < 4 (have: $(cmake --version 2>/dev/null | head -1))." >&2
        echo "       Install without admin:  uv tool install 'cmake<4'" >&2
        exit 1
    fi
    if [ "$pyminor" -ge 13 ]; then
        echo "ERROR: OpenCV's Obj-C binding generator needs python 3.12 (have: $(python3 --version 2>/dev/null))." >&2
        echo "       Install:  uv python install 3.12" >&2
        exit 1
    fi
    echo "Toolchain OK: $(cmake --version | head -1), $(python3 --version)"
}

COMMON_ARGS=(
    --without dnn --without gapi --without highgui --without ml
    --without objdetect --without photo --without stitching
    --without video --without videoio --without parallel
    --without ptcloud  # unused; its objc VolumeType binding clashes with macOS Carbon OSType
    --macos_archs=arm64 --iphoneos_archs=arm64 --iphonesimulator_archs=arm64
    --catalyst_archs='' --disable-bitcode --build_only_specified_archs
)
[ "$DYNAMIC" = "1" ] && COMMON_ARGS+=(--dynamic)

# iOS frameworks must be shallow bundles (Info.plist + binary at the root). Static
# frameworks build deep/versioned, so flatten the iOS slices (macOS stays versioned).
flatten_ios_slices() {
    local xc="$1" slice fw
    for slice in ios-arm64 ios-arm64-simulator; do
        fw="$xc/$slice/opencv2.framework"
        [ -d "$fw/Versions" ] || continue
        rm -f "$fw/Headers" "$fw/Modules" "$fw/Resources" "$fw/opencv2"
        mv "$fw/Versions/A/opencv2" "$fw/Versions/A/Headers" "$fw/Versions/A/Modules" "$fw/"
        mv "$fw/Versions/A/Resources/Info.plist" "$fw/Info.plist"
        rm -rf "$fw/Versions"
    done
}

# Convert a static .framework xcframework into a static-LIBRARY xcframework
# (libopencv2.a + Headers/opencv2) so SPM links it instead of embedding it.
repackage_static_lib() {
    local fwxc="$1" outxc="$2" strip_dbg="${3:-no}" stage="$FRAMEWORKS_PATH/.libstage"
    rm -rf "$stage" "$outxc"; mkdir -p "$stage"
    local args=() slice fw bin hdr
    for slice in ios-arm64 ios-arm64-simulator macos-arm64; do
        fw="$fwxc/$slice/opencv2.framework"; [ -d "$fw" ] || continue
        if [ -d "$fw/Versions/A" ]; then bin="$fw/Versions/A/opencv2"; hdr="$fw/Versions/A/Headers"
        else bin="$fw/opencv2"; hdr="$fw/Headers"; fi
        mkdir -p "$stage/$slice/include/opencv2"
        cp "$bin" "$stage/$slice/libopencv2.a"
        # DWARF in a static .a isn't linked into the app (it doesn't change app/App-Clip
        # binary size) but it bloats the repo ~7x; strip it for the distribution variant.
        [ "$strip_dbg" = "yes" ] && strip -S "$stage/$slice/libopencv2.a"
        cp -R "$hdr"/. "$stage/$slice/include/opencv2/"
        args+=(-library "$stage/$slice/libopencv2.a" -headers "$stage/$slice/include")
    done
    xcodebuild -create-xcframework "${args[@]}" -output "$outxc"
    rm -rf "$stage"
}

# build_variant <final-name> [extra build_xcframework.py args...]
build_variant() {
    local final="$1" strip_dbg="$2"; shift 2
    echo "==> Building $final"
    rm -rf "$FRAMEWORKS_PATH/opencv2.xcframework" "$FRAMEWORKS_PATH/$final"
    python3 "$OPENCV_BUILD" --out "$FRAMEWORKS_PATH" "${COMMON_ARGS[@]}" "$@"
    flatten_ios_slices "$FRAMEWORKS_PATH/opencv2.xcframework"
    if [ "$DYNAMIC" = "1" ]; then
        mv "$FRAMEWORKS_PATH/opencv2.xcframework" "$FRAMEWORKS_PATH/$final"   # dynamic framework, embeddable
    else
        repackage_static_lib "$FRAMEWORKS_PATH/opencv2.xcframework" "$FRAMEWORKS_PATH/$final" "$strip_dbg"  # static lib, linked
        rm -rf "$FRAMEWORKS_PATH/opencv2.xcframework"
    fi
    echo "==> Done: $FRAMEWORKS_PATH/$final"
}

ensure_toolchain
"$ROOT/script/patches/apply_opencv5_patches.sh"
mkdir -p "$FRAMEWORKS_PATH"

if [ "$WHICH" = "debug" ] || [ "$WHICH" = "both" ]; then
    build_variant opencv2-debug.xcframework no --debug_info   # keep opencv debug symbols
fi
if [ "$WHICH" = "optimized" ] || [ "$WHICH" = "both" ]; then
    build_variant opencv2-optimized.xcframework yes           # strip DWARF (smaller repo)
fi

# Install: the debug variant becomes the default opencv2.xcframework Package.swift links;
# the optimized variant keeps its suffixed name (selected via OPENCV_OPTIMIZED=1).
mkdir -p "$ROOT/lib"
if [ -d "$FRAMEWORKS_PATH/opencv2-debug.xcframework" ]; then
    rm -rf "$ROOT/lib/opencv2.xcframework"
    mv "$FRAMEWORKS_PATH/opencv2-debug.xcframework" "$ROOT/lib/opencv2.xcframework"
    echo "Installed $ROOT/lib/opencv2.xcframework (debug)"
fi
if [ -d "$FRAMEWORKS_PATH/opencv2-optimized.xcframework" ]; then
    rm -rf "$ROOT/lib/opencv2-optimized.xcframework"
    mv "$FRAMEWORKS_PATH/opencv2-optimized.xcframework" "$ROOT/lib/opencv2-optimized.xcframework"
    echo "Installed $ROOT/lib/opencv2-optimized.xcframework"
fi
