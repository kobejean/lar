include(ExternalProject)
include(FetchContent)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

set(DEPENDENCIES)
set(EXTRA_CMAKE_ARGS)

# Platform-specific toolchain settings
if(APPLE)
  set(COMMON_TOOLCHAIN_ARGS
    -DCMAKE_C_COMPILER=/usr/bin/clang
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    -DCMAKE_AR=/usr/bin/ar
    -DCMAKE_RANLIB=/usr/bin/ranlib
  )
elseif(UNIX AND NOT APPLE)
  # Linux settings - use default system compilers
  set(COMMON_TOOLCHAIN_ARGS
    -DCMAKE_C_COMPILER=gcc
    -DCMAKE_CXX_COMPILER=g++
  )
else()
  # Windows or other platforms - use CMake defaults
  set(COMMON_TOOLCHAIN_ARGS "")
endif()

# Setup Eigen
list(APPEND DEPENDENCIES Eigen3)
ExternalProject_Add(Eigen3
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/eigen3
  BINARY_DIR Eigen3-build
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
    ${COMMON_TOOLCHAIN_ARGS}
)
list(APPEND EXTRA_CMAKE_ARGS
  -DEigen3_DIR=${CMAKE_BINARY_DIR}/Eigen3-build
  -DEIGEN3_INCLUDE_DIR=${CMAKE_BINARY_DIR}/install/include/eigen3
)

# Setup OpenCV
list(APPEND DEPENDENCIES opencv)
ExternalProject_Add(opencv
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/opencv
  BINARY_DIR opencv-build
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
    ${COMMON_TOOLCHAIN_ARGS}
    -DBUILD_LIST=core,calib3d,features2d,imgcodecs,imgproc
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_DOCS=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_opencv_apps=OFF
    -DBUILD_opencv_world=OFF
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
    -DBUILD_JPEG_TURBO_DISABLE=ON
    -DBUILD_ZLIB=OFF
    -DWITH_ZLIB=ON
)
list(APPEND EXTRA_CMAKE_ARGS -DOpenCV_DIR=${CMAKE_BINARY_DIR}/opencv-build)

# Setup nlohmann_json
list(APPEND DEPENDENCIES nlohmann_json)
ExternalProject_Add(nlohmann_json
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/json
    BINARY_DIR nlohmann_json-build
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
        ${COMMON_TOOLCHAIN_ARGS}
        -DJSON_BuildTests=OFF
        -DJSON_Install=ON
        -DJSON_MultipleHeaders=OFF
        -DBUILD_TESTING=OFF
        -DJSON_ImplicitConversions=ON
        -DJSON_Diagnostics=OFF
        -DJSON_SystemInclude=OFF
    LOG_CONFIGURE ON
    LOG_BUILD ON
)
list(APPEND EXTRA_CMAKE_ARGS -Dnlohmann_json_DIR=${CMAKE_BINARY_DIR}/nlohmann_json-build)

# Setup g2o
list(APPEND DEPENDENCIES g2o)
ExternalProject_Add(g2o
  DEPENDS Eigen3
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/g2o
  BINARY_DIR g2o-build
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
    -DEIGEN3_INCLUDE_DIR=${CMAKE_BINARY_DIR}/install/include/eigen3
    ${COMMON_TOOLCHAIN_ARGS}
    -DG2O_USE_VENDORED_CERES=ON
    -DG2O_USE_OPENGL=ON
    -DBUILD_SHARED_LIBS=OFF
    -DG2O_BUILD_APPS=OFF
    -DG2O_BUILD_EXAMPLES=OFF
    -DG2O_BUILD_SLAM2D_TYPES=OFF
    -DG2O_BUILD_SLAM2D_ADDON_TYPES=OFF
    -DG2O_BUILD_DATA_TYPES=OFF
    -DG2O_BUILD_SCLAM2D_TYPES=OFF
    -DG2O_BUILD_ICP_TYPES=OFF
    -DG2O_BUILD_SIM3_TYPES=OFF
)
list(APPEND EXTRA_CMAKE_ARGS
  -Dg2o_DIR=${CMAKE_BINARY_DIR}/install/lib/cmake/g2o
  -DG2O_USE_VENDORED_CERES=ON
  -DG2O_USE_OPENGL=ON
  -DBUILD_SHARED_LIBS=OFF
  ${COMMON_TOOLCHAIN_ARGS}
)

# Inner build
ExternalProject_Add(ep_lar
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS ${CL_ARGS} -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lar
  BUILD_ALWAYS ON
)