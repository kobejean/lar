mkdir tmp
cd tmp

mkdir install
cmake ../thirdparty/eigen3 -DCMAKE_INSTALL_PREFIX=./install
make install -j 8
cp -R ./install/include/eigen3 ../include
rm -r *

mkdir install
cmake ../thirdparty/opencv -DCMAKE_INSTALL_PREFIX=./install \
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
    -DWITH_TIFF=ON \
    -DWITH_WEBP=OFF \
    -DWITH_OPENJPEG=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_IMGCODEC_HDR=OFF \
    -DWITH_IMGCODEC_SUNRASTER=OFF \
    -DWITH_IMGCODEC_PXM=OFF \
    -DWITH_IMGCODEC_PFM=ON
make install -j 8
cp -R ./install/include/opencv4/opencv2 ../include
rm -r *

mkdir install
cmake ../thirdparty/g2o -DCMAKE_INSTALL_PREFIX=./install -DEIGEN3_INCLUDE_DIR=$(cd "../include/eigen3"; pwd) -DG2O_USE_VENDORED_CERES=ON
make install -j 8
cp -R ./install/include/g2o ../include
rm -r *

cd ..
make all