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
set(WITH_TIFF OFF)
set(WITH_WEBP OFF)
set(WITH_OPENJPEG OFF)
set(WITH_JASPER OFF)
set(WITH_OPENEXR OFF)
set(WITH_IMGCODEC_HDR OFF)
set(WITH_IMGCODEC_SUNRASTER OFF)
set(WITH_IMGCODEC_PXM OFF)
set(WITH_IMGCODEC_PFM ON)

add_subdirectory(thirdparty/opencv)

# Setup G2O
add_definitions(-DG2O_USE_VENDORED_CERES) # TODO: Figure out why we have specify this
add_definitions(-DG2O_BUILD_APPS=0)
add_definitions(-DG2O_BUILD_EXAMPLES=0)
add_definitions(-DG2O_BUILD_SLAM2D_TYPES=0)
add_definitions(-DG2O_BUILD_SLAM2D_ADDON_TYPES=0)
add_definitions(-DG2O_BUILD_DATA_TYPES=0)
add_definitions(-DG2O_BUILD_SCLAM2D_TYPES=0)
add_definitions(-DG2O_BUILD_ICP_TYPES=0)
add_definitions(-DG2O_BUILD_SIM3_TYPES=0)
add_subdirectory(thirdparty/g2o)

# Setup nlohmann_json
add_subdirectory(thirdparty/json)

# Add subdirectories
add_subdirectory(apps)
add_subdirectory(src)

if(GEOAR_BUILD_TESTS)
  message("GeoAR Build Tests Enabled")
  enable_testing()
  add_subdirectory(test)
endif()

set_target_properties(
  g2o_cli_application
  static_target
  static_dynamic_function_fit
  constant_velocity_target
  ba_demo
  bal_example
  sclam_odom_laser
  sclam_laser_calib
  curve_fit
  convert_sba_slam3d
  sba_demo
  ba_anchored_inverse_depth_demo
  sclam_pure_calibration
  optimize_sphere_by_sim3
  gicp_sba_demo
  g2o_hierarchical_application
  create_sphere
  tutorial_slam2d
  simulator_3d_plane
  g2o_cli_application
  gicp_demo
  polynomial_fit
  simulator_3d_line
  circle_fit
  simple_optimize
  g2o_cli_library
  
  PROPERTIES EXCLUDE_FROM_ALL ON
)