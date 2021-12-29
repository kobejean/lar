
include (ExternalProject)
include(FetchContent)

set (DEPENDENCIES)
set (EXTRA_CMAKE_ARGS)


# Setup Eigen

list (APPEND DEPENDENCIES Eigen3)
ExternalProject_Add(Eigen3
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/eigen3
  BINARY_DIR Eigen3-build
  # INSTALL_COMMAND ""
  CMAKE_ARGS 
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/Eigen3-install
)
list(APPEND EXTRA_CMAKE_ARGS
  -DEigen3_DIR=${CMAKE_BINARY_DIR}/Eigen3-build
  -DEIGEN3_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/Eigen3-install/include/eigen3
)


# Setup OpenCV

list (APPEND DEPENDENCIES opencv)
ExternalProject_Add(opencv
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/opencv
  BINARY_DIR opencv-build
  INSTALL_COMMAND ""
  CMAKE_ARGS 
    # -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/thirdparty/install
    -DBUILD_LIST=core,calib3d,features2d,imgcodecs,imgproc
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_DOCS=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_opencv_apps=OFF
    -DBUILD_PNG=OFF
    -DBUILD_TIFF=OFF
    -DBUILD_WEBP=OFF
    -DBUILD_OPENJPEG=OFF
    -DBUILD_JASPER=OFF
    -DBUILD_OPENEXR=OFF
    -DWITH_PNG=OFF
    -DWITH_JPEG=ON
    -DWITH_TIFF=OFF
    -DWITH_WEBP=OFF
    -DWITH_OPENJPEG=OFF
    -DWITH_JASPER=OFF
    -DWITH_OPENEXR=OFF
    -DWITH_IMGCODEC_HDR=OFF
    -DWITH_IMGCODEC_SUNRASTER=OFF
    -DWITH_IMGCODEC_PXM=OFF
    -DWITH_IMGCODEC_PFM=ON
)
list(APPEND EXTRA_CMAKE_ARGS -DOpenCV_DIR=${CMAKE_BINARY_DIR}/opencv-build)


# Setup nlohmann_json

list (APPEND DEPENDENCIES nlohmann_json)
ExternalProject_Add(nlohmann_json
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/json
  BINARY_DIR nlohmann_json-build
  INSTALL_COMMAND ""
  # CMAKE_ARGS 
  #   # -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/thirdparty/install
  #   -DBUILD_LIST=core,calib3d,features2d,imgcodecs,imgproc
)
list(APPEND EXTRA_CMAKE_ARGS -Dnlohmann_json_DIR=${CMAKE_BINARY_DIR}/nlohmann_json-build)

ExternalProject_Add(ep_geoar
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS ${CL_ARGS} -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/geoar
  BUILD_ALWAYS ON
)