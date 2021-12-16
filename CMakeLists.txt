cmake_minimum_required(VERSION 3.14)
project(geoar)
include(FetchContent)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)

# Generate third party header include files if it doesn't exist
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/include")
  message("Making Third Party Includes")
  execute_process(COMMAND "${PROJECT_SOURCE_DIR}/script/build_thirdparty_includes.bash")
endif()

# Setup output directories
file(MAKE_DIRECTORY lib)
file(MAKE_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/include)


# Setup Eigen
add_subdirectory(thirdparty/eigen3)
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# Setup OpenCV

set(BUILD_LIST "calib3d,features2d,imgcodecs")
set(BUILD_SHARED_LIBS OFF)
set(BUILD_DOCS OFF)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_opencv_apps OFF)
set(BUILD_PNG OFF)
set(BUILD_TIFF OFF)
set(BUILD_WEBP OFF)
set(BUILD_OPENJPEG OFF)
set(BUILD_JASPER OFF)
set(BUILD_OPENEXR OFF)
set(WITH_PNG OFF)
set(WITH_JPEG ON)
set(WITH_TIFF ON)
set(WITH_WEBP OFF)
set(WITH_OPENJPEG OFF)
set(WITH_JASPER OFF)
set(WITH_OPENEXR OFF)
set(WITH_IMGCODEC_HDR OFF)
set(WITH_IMGCODEC_SUNRASTER OFF)
set(WITH_IMGCODEC_PXM OFF)
set(WITH_IMGCODEC_PFM ON)

add_subdirectory(thirdparty/opencv)
list(APPEND OpenCV_LIBS opencv_core opencv_features2d opencv_imgcodecs)

# Setup G2O
add_definitions(-DG2O_USE_VENDORED_CERES) # TODO: Figure out why we have specify this
add_subdirectory(thirdparty/g2o)

# Create G2O_LIBS list
list(APPEND G2O_LIBS types_sba types_slam3d solver_dense)
list(APPEND G2O_LIBS solver_eigen)

# Setup nlohmann_json
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.10.2
)
FetchContent_MakeAvailable(json)

# Add subdirectories
add_subdirectory(apps)
add_subdirectory(src)

if(GEOAR_BUILD_TESTS)
  message("GeoAR Build Tests Enabled")
  enable_testing()
  add_subdirectory(test)
endif()
