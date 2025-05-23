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
  project(lar)
endif()

# LAR

include(FetchContent)
include(ExternalProject)

# Fix for macOS compilation issues - must be set before any target compilation
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-macro-redefined -include alloca.h")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-macro-redefined -include alloca.h")
    add_definitions(-D_POSIX_C_SOURCE=200809L)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)

# Options
option(LAR_BUILD_APPS "Whether or not apps should be built" ON)
option(LAR_BUILD_TESTS "Whether or not tests should be built" OFF)
option(LAR_COMPACT_BUILD "Build the bare minimum" OFF)

if(LAR_COMPACT_BUILD)
  # TODO: Figure out what compact build should cover
  # message("LAR Compact Build Enabled")
  # add_definitions(-DLAR_COMPACT_BUILD)
endif()

# Setup output directories
file(MAKE_DIRECTORY lib)
file(MAKE_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 3.4.90 REQUIRED)
find_package(OpenCV 4.5.4 REQUIRED core calib3d features2d imgcodecs imgproc)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
if(LAR_BUILD_APPS AND NOT LAR_COMPACT_BUILD)
  message("LAR Build Apps Enabled")
  add_subdirectory(apps)
endif()

add_subdirectory(src)

if(LAR_BUILD_TESTS AND NOT LAR_COMPACT_BUILD)
  message("LAR Build Tests Enabled")
  enable_testing()
  add_subdirectory(test)
endif()