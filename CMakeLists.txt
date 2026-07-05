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

# Per-function/data sections + linker dead-stripping so that code from unreferenced
# modules (e.g. image I/O in a tracking-only binary) is dropped from the final executable.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
elseif(UNIX)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

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
find_package(OpenCV 5.0 REQUIRED core geometry features flann imgproc)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

# gRPC and Protobuf (optional, for service build)
option(LAR_BUILD_SERVICE "Whether or not the gRPC service should be built" ON)
if(LAR_BUILD_SERVICE)
  # Try CONFIG mode first (Homebrew/vcpkg), fall back to MODULE mode (Ubuntu apt)
  # Must use CONFIG on macOS to avoid conflicts with gRPC's find_dependency
  find_package(Protobuf CONFIG QUIET)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf MODULE REQUIRED)
  endif()
  find_package(gRPC CONFIG REQUIRED)

  # FindZLIB on macOS resolves ZLIB_INCLUDE_DIR to the SDK's usr/include, which
  # protobuf/gRPC propagate transitively via ZLIB::ZLIB. Passed as an explicit
  # -isystem <SDK>/usr/include, that dir lands ahead of libc++'s c++/v1 headers
  # and breaks #include_next (the "<cstddef> tried including <stddef.h> but didn't
  # find libc++'s <stddef.h>" cascade). CMake only auto-strips this when
  # CMAKE_OSX_SYSROOT is set; it isn't here. It's a compiler-implicit dir, so
  # remove it from the interface includes — zlib.h is still found via the default
  # SDK search.
  if(APPLE AND TARGET ZLIB::ZLIB)
    get_target_property(_zlib_incs ZLIB::ZLIB INTERFACE_INCLUDE_DIRECTORIES)
    if(_zlib_incs)
      list(FILTER _zlib_incs EXCLUDE REGEX "MacOSX\\.sdk/usr/include$")
      set_target_properties(ZLIB::ZLIB PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_zlib_incs}")
    endif()
  endif()

  # Proto compilation setup
  set(PROTO_DIR ${PROJECT_SOURCE_DIR}/proto)
  set(PROTO_GEN_DIR ${PROJECT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

  # Find protoc - handle both CONFIG mode (protobuf::protoc) and MODULE mode (Protobuf_PROTOC_EXECUTABLE)
  if(TARGET protobuf::protoc)
    get_target_property(PROTOC_EXECUTABLE protobuf::protoc LOCATION)
  else()
    set(PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
  endif()
  get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

  # Generate proto sources
  set(PROTO_SRCS ${PROTO_GEN_DIR}/navigation.pb.cc)
  set(PROTO_HDRS ${PROTO_GEN_DIR}/navigation.pb.h)
  set(GRPC_SRCS ${PROTO_GEN_DIR}/navigation.grpc.pb.cc)
  set(GRPC_HDRS ${PROTO_GEN_DIR}/navigation.grpc.pb.h)

  add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_EXECUTABLE}
      --cpp_out=${PROTO_GEN_DIR}
      --grpc_out=${PROTO_GEN_DIR}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
      -I${PROTO_DIR}
      ${PROTO_DIR}/navigation.proto
    DEPENDS ${PROTO_DIR}/navigation.proto
    COMMENT "Generating gRPC/protobuf sources"
  )

  # Create proto library - handle both CONFIG and MODULE mode targets
  add_library(lar_proto STATIC ${PROTO_SRCS} ${GRPC_SRCS})
  if(TARGET protobuf::libprotobuf)
    target_link_libraries(lar_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
  else()
    target_link_libraries(lar_proto PUBLIC ${Protobuf_LIBRARIES} gRPC::grpc++)
    target_include_directories(lar_proto PUBLIC ${Protobuf_INCLUDE_DIRS})
  endif()
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