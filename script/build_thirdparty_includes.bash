#!/usr/bin/env bash
PROJECT_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
INCLUDE_DIR="$PROJECT_ROOT_DIR/thirdparty/include"
TMP_DIR="$PROJECT_ROOT_DIR/tmp"
TMP_INSTALL_DIR="$TMP_DIR/install"

mkdir $INCLUDE_DIR
mkdir $TMP_DIR
cd $TMP_DIR

mkdir $TMP_INSTALL_DIR
cmake "$PROJECT_ROOT_DIR/thirdparty/json" -DCMAKE_INSTALL_PREFIX=$TMP_INSTALL_DIR
make install -j 8
EIGEN_STATUS=$?
cp -R "$TMP_INSTALL_DIR/include/nlohmann" $INCLUDE_DIR
rm -r *

mkdir $TMP_INSTALL_DIR
cmake "$PROJECT_ROOT_DIR/thirdparty/eigen3" -DCMAKE_INSTALL_PREFIX=$TMP_INSTALL_DIR
make install -j 8
EIGEN_STATUS=$?
cp -R "$TMP_INSTALL_DIR/include/eigen3" $INCLUDE_DIR
rm -r *

mkdir $TMP_INSTALL_DIR
cmake "$PROJECT_ROOT_DIR/thirdparty/opencv" -DCMAKE_INSTALL_PREFIX=$TMP_INSTALL_DIR \
    -DBUILD_LIST="calib3d,features2d,imgcodecs" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_PNG=OFF \
    -DBUILD_TIFF=OFF \
    -DBUILD_WEBP=OFF \
    -DBUILD_OPENJPEG=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DWITH_PNG=OFF \
    -DWITH_JPEG=ON \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_OPENJPEG=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_IMGCODEC_HDR=OFF \
    -DWITH_IMGCODEC_SUNRASTER=OFF \
    -DWITH_IMGCODEC_PXM=OFF \
    -DWITH_IMGCODEC_PFM=ON
make install -j 8
OPENCV_STATUS=$?
cp -R "$TMP_INSTALL_DIR/include/opencv4/opencv2" $INCLUDE_DIR
rm -r *

mkdir install
cmake "$PROJECT_ROOT_DIR/thirdparty/g2o" -DCMAKE_INSTALL_PREFIX=$TMP_INSTALL_DIR -DEIGEN3_INCLUDE_DIR="$INCLUDE_DIR/eigen3" -DG2O_USE_VENDORED_CERES=ON
make install -j 8
G2O_STATUS=$?
cp -R "$TMP_INSTALL_DIR/include/g2o" $INCLUDE_DIR
rm -r *

if [ $EIGEN_STATUS -eq 0 ] && [ $OPENCV_STATUS -eq 0 ] && [ $G2O_STATUS -eq 0 ]
then
    echo "Third Party Includes Successfully Generated"
    rm -rf $TMP_DIR
else
    cd $PROJECT_ROOT_DIR
    rm -rf $INCLUDE_DIR $TMP_DIR
fi
