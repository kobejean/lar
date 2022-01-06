cmake_minimum_required(VERSION 3.14)

# Get command line cmake args. MUST be done before call to 'project'
get_cmake_property(vars CACHE_VARIABLES)
foreach(var ${vars})
  get_property(currentHelpString CACHE "${var}" PROPERTY HELPSTRING)
  if("${currentHelpString}" MATCHES "No help, variable specified on the command line." OR "${currentHelpString}" STREQUAL "")
    # message("${var} = [${${var}}]  --  ${currentHelpString}") # uncomment to see the variables being processed
    list(APPEND CL_ARGS "-D${var}=${${var}}")
  endif()
endforeach()

# Super Build
option(USE_SUPERBUILD "Whether or not a superbuild should be invoked" ON)
if(USE_SUPERBUILD)
  project(SUPERBUILD NONE)
  include(cmake/SuperBuild.cmake)
  return()
else()
  project(geoar)
endif()


# GeoAR

include(FetchContent)
include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)

# Options
option(GEOAR_BUILD_APPS "Whether or not apps should be built" ON)
option(GEOAR_BUILD_TESTS "Whether or not tests should be built" OFF)

# Setup output directories
file(MAKE_DIRECTORY lib)
file(MAKE_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 3.4.90 REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED core calib3d features2d imgcodecs imgproc)
find_package(nlohmann_json 3.10.4 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
if(GEOAR_BUILD_APPS)
  message("GeoAR Build Apps Enabled")
  add_subdirectory(apps)
endif()

add_subdirectory(src)

if(GEOAR_BUILD_TESTS)
  message("GeoAR Build Tests Enabled")
  enable_testing()
  add_subdirectory(test)
endif()