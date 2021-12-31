cmake_minimum_required(VERSION 3.14)

# MUST be done before call to 'project'
get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  get_property(currentHelpString CACHE "${var}" PROPERTY HELPSTRING)
  if("${currentHelpString}" MATCHES "No help, variable specified on the command line." OR "${currentHelpString}" STREQUAL "")
    # message("${var} = [${${var}}]  --  ${currentHelpString}") # uncomment to see the variables being processed
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

# Super Build
option (USE_SUPERBUILD "Whether or not a superbuild should be invoked" ON)
if (USE_SUPERBUILD)
  project(SUPERBUILD NONE)
  include(cmake/SuperBuild.cmake)
  return()                      # stop processing this file further
else()
  project(geoar)
endif()


# GeoAR

include(FetchContent)
include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)

# Setup output directories
file(MAKE_DIRECTORY lib)
file(MAKE_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED core calib3d features2d imgcodecs imgproc)
find_package(nlohmann_json REQUIRED)
find_package(G2O REQUIRED)
find_package(OpenGL)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Setup G2O

include(${G2O_DIR}/g2oTargets.cmake)
add_definitions(-DG2O_USE_VENDORED_CERES)
add_definitions(-DG2O_HAVE_OPENGL=OFF)

# Add subdirectories
add_subdirectory(apps)
add_subdirectory(src)

if(GEOAR_BUILD_TESTS)
  message("GeoAR Build Tests Enabled")
  enable_testing()
  add_subdirectory(test)
endif()