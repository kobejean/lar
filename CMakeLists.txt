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

# gRPC and Protobuf (optional, for service build)
option(LAR_BUILD_SERVICE "Whether or not the gRPC service should be built" ON)
if(LAR_BUILD_SERVICE)
  find_package(protobuf CONFIG REQUIRED)
  find_package(gRPC CONFIG REQUIRED)

  # Proto compilation setup
  set(PROTO_DIR ${PROJECT_SOURCE_DIR}/proto)
  set(PROTO_GEN_DIR ${PROJECT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

  # Find protoc and grpc_cpp_plugin
  get_target_property(PROTOC_EXECUTABLE protobuf::protoc LOCATION)
  get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

  # Generate proto sources
  set(PROTO_SRCS ${PROTO_GEN_DIR}/pathfinding.pb.cc)
  set(PROTO_HDRS ${PROTO_GEN_DIR}/pathfinding.pb.h)
  set(GRPC_SRCS ${PROTO_GEN_DIR}/pathfinding.grpc.pb.cc)
  set(GRPC_HDRS ${PROTO_GEN_DIR}/pathfinding.grpc.pb.h)

  add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_EXECUTABLE}
      --cpp_out=${PROTO_GEN_DIR}
      --grpc_out=${PROTO_GEN_DIR}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
      -I${PROTO_DIR}
      ${PROTO_DIR}/pathfinding.proto
    DEPENDS ${PROTO_DIR}/pathfinding.proto
    COMMENT "Generating gRPC/protobuf sources"
  )

  # Create proto library
  add_library(lar_proto STATIC ${PROTO_SRCS} ${GRPC_SRCS})
  target_link_libraries(lar_proto PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
  )
  target_include_directories(lar_proto PUBLIC ${PROTO_GEN_DIR})
endif()

# Add g2o library directory to linker search path
if(g2o_DIR)
    get_filename_component(G2O_LIB_DIR "${g2o_DIR}/../../" ABSOLUTE)
    link_directories(${G2O_LIB_DIR})
endif()

# Find OpenGL for g2o visualization support
if(APPLE)
    find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)
    find_library(GLUT_FRAMEWORK GLUT)
    set(OPENGL_LIBRARIES ${OPENGL_FRAMEWORK})
    if(GLUT_FRAMEWORK)
        list(APPEND OPENGL_LIBRARIES ${GLUT_FRAMEWORK})
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux - use standard OpenGL packages
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif()

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