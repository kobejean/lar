mkdir buildf
# cd build
    # "-DCMAKE_OSX_ARCHITECTURES=armv7;armv7s;arm64;i386;x86_64" \
    # -DOPENCV_3P_LIB_INSTALL_PATH=lib/3rdparty \
cmake -S. -Bbuildf -GXcode -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=Q74A2VY23K \
    -DUSE_SUPERBUILD=OFF \
    -DOpenCV_DIR=`pwd`/build/opencv-build \
    -DEigen3_DIR=`pwd`/build/Eigen3-build \
    -DEIGEN3_INCLUDE_DIR=`pwd`/build/Eigen3-install/include/eigen3 \
    -Dnlohmann_json_DIR=`pwd`/build/nlohmann_json-build \
    "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.0 \
    -DCMAKE_INSTALL_PREFIX=`pwd`/_install \
    -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
    -DAPPLE_FRAMEWORK=ON \
    -DCMAKE_IOS_INSTALL_COMBINED=YES
cmake --build buildf --config Release --target install --verbose