mkdir build/framework
# cd build
    # "-DCMAKE_OSX_ARCHITECTURES=armv7;armv7s;arm64;i386;x86_64" \
    # -DOPENCV_3P_LIB_INSTALL_PATH=lib/3rdparty \
cmake -S. -Bbuild/framework -GXcode -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=Q74A2VY23K \
    -DUSE_SUPERBUILD=OFF \
    -DOpenCV_DIR=`pwd`/build/opencv-build \
    -DEigen3_DIR=`pwd`/build/Eigen3-build \
    -DEIGEN3_INCLUDE_DIR=`pwd`/build/Eigen3-install/include/eigen3 \
    -Dnlohmann_json_DIR=`pwd`/build/nlohmann_json-build \
    -Dg2o_DIR=`pwd`/build/g2o-install/lib/cmake/g2o \
    -DG2O_USE_VENDORED_CERES=ON \
    -DG2O_USE_OPENGL=OFF \
    "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
    -DCMAKE_INSTALL_PREFIX=`pwd`/_install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DAPPLE_FRAMEWORK=ON \
    -DCMAKE_IOS_INSTALL_COMBINED=YES
cmake --build build/framework --config Release --target geoar_process --verbose