cmake_minimum_required(VERSION 3.14)
project(geoar)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
set(CMAKE_CXX_STANDARD 14)

# Setup output directories
file(MAKE_DIRECTORY lib)
file(MAKE_DIRECTORY bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Add subdirectories
add_subdirectory(apps)
add_subdirectory(src)


# Setup Eigen
find_package(Eigen3 3.3 REQUIRED)
if(TARGET Eigen3::Eigen)
  set(G2O_EIGEN3_EIGEN_TARGET Eigen3::Eigen)
else()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Setup G2O
FetchContent_Declare(
  g2o
  GIT_REPOSITORY https://github.com/RainerKuemmerle/g2o.git
  GIT_TAG 5b14535b8d63f837699d1dde27fd350fc6738b2a
)
FetchContent_MakeAvailable(g2o)
include_directories(${g2o_SOURCE_DIR} ${g2o_BINARY_DIR})
link_directories(${g2o_SOURCE_DIR}/lib)
add_definitions(-DG2O_USE_VENDORED_CERES) # TODO: Figure out why we have specify this
# Create G2O_LIBS list
list(APPEND G2O_LIBS g2o_core g2o_types_sba g2o_types_slam3d g2o_solver_dense)
if(CHOLMOD_FOUND)
  list(APPEND G2O_LIBS g2o_solver_cholmod)
else()
  list(APPEND G2O_LIBS g2o_solver_eigen)
endif()


# Setup nlohmann_json
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.10.2
)
FetchContent_MakeAvailable(json)
link_libraries(nlohmann_json::nlohmann_json)

