#!/usr/bin/env bash
#
# Apply the OpenCV 5.0.0 source workarounds needed to build its iOS xcframework
# under Xcode 26. These patch the vendored thirdparty/opencv (5.0.0 tag).
#
# All are upstream-known and fixed on the 5.x branch but not in the 5.0.0 release,
# so DELETE this script once thirdparty/opencv is bumped to a 5.0.x that includes:
#   - PR #29257  (LightGlueMatcher swift_name)
#   - the Xcode-26 docbuild / tapi-version fix
#
# Idempotent: safe to run repeatedly. Invoked by build_opencv_framework.bash.
#
# NOTE: the toolchain requirements live in build_opencv_framework.bash, not here:
#   - cmake < 4   (OpenCV 4/5 -GXcode compiler detection breaks on cmake 4.x)
#   - python 3.12 (OpenCV's Obj-C binding generator breaks on python 3.14)
set -euo pipefail

OPENCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../thirdparty/opencv" && pwd)"
echo "Applying OpenCV 5 build workarounds to $OPENCV"

# 1) build_framework.py's get_current_branch() runs `git branch --show-current`,
#    which is empty on a detached HEAD (we check out the 5.0.0 tag) -> hard error.
#    Put opencv on a named branch at the same commit.
if ! git -C "$OPENCV" symbolic-ref -q HEAD >/dev/null 2>&1; then
    rev="$(git -C "$OPENCV" rev-parse --short HEAD)"
    git -C "$OPENCV" switch -c "opencv-build-${rev}" 2>/dev/null \
        || git -C "$OPENCV" switch "opencv-build-${rev}"
    echo "  [1/3] opencv on branch opencv-build-${rev}"
else
    echo "  [1/3] opencv already on a named branch"
fi

# 2) LightGlueMatcher's Obj-C binding emits a conflicting swift_name (clang rejects
#    it as a hard error). Upstream fix PR #29257: rename the create(modelPath)
#    factory to createFromFile via a gen_dict.json func_arg_fix.
python3 - "$OPENCV/modules/features/misc/objc/gen_dict.json" <<'PY'
import collections, json, sys
path = sys.argv[1]
data = json.load(open(path), object_pairs_hook=collections.OrderedDict)
faf = data.setdefault("func_arg_fix", collections.OrderedDict())
sig = ("(LightGlueMatcher*)create:(NSString*)modelPath scoreThreshold:(float)"
       "scoreThreshold backend:(int)backend target:(int)target")
lg = faf.setdefault("LightGlueMatcher", collections.OrderedDict())
lg[sig] = {"create": {"name": "createFromFile"}}
json.dump(data, open(path, "w"), indent=4)
PY
echo "  [2/3] LightGlueMatcher create -> createFromFile (PR #29257)"

# 3) DocC docbuild step fails on Xcode 26 ("Could not parse tapi version"); we
#    don't ship docs, so disable it (it is force-enabled for Xcode >= 13).
perl -0pi -e \
    's/self\.build_docs = xcode_ver >= 13/self.build_docs = False  # Xcode 26 docbuild tapi bug/' \
    "$OPENCV/platforms/ios/build_framework.py"
echo "  [3/4] DocC docbuild disabled (ios/build_framework.py)"

# 4) With docbuild disabled there is no docs/ dir, so build_xcframework.py's
#    "copy documentation" phase must tolerate its absence.
perl -0pi -e \
    's{(docs_dst = "\{\}/docs_\{\}"\.format\(args\.out, platform\)\n)}{$1            if not os.path.exists(docs_src):\n                continue\n}' \
    "$OPENCV/platforms/apple/build_xcframework.py"
echo "  [4/4] docs-copy made tolerant of missing docs (apple/build_xcframework.py)"

echo "Done."
