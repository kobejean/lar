cmake_minimum_required(VERSION 3.0)
project(GeoARCore)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
set(CMAKE_CXX_STANDARD 14)

# Setup Eigen
find_package(Eigen3 3.3 REQUIRED)
if (TARGET Eigen3::Eigen)
  set(G2O_EIGEN3_EIGEN_TARGET Eigen3::Eigen)
else ()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif ()

# Setup G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
link_directories(${G2O_LIB_DIR})
add_definitions(-DG2O_USE_VENDORED_CERES) # TODO: Figure out why we have specify this
# Create G2O_LIBS list
list(APPEND G2O_LIBS g2o_core g2o_types_sba g2o_types_slam3d g2o_solver_dense)
if (CHOLMOD_FOUND)
  list(APPEND G2O_LIBS g2o_solver_cholmod)
else ()
  list(APPEND G2O_LIBS g2o_solver_eigen)
endif ()



# Create ba_demo executable
add_executable(ba_demo src/demo/ba_demo.cpp)
target_link_libraries(ba_demo ${G2O_LIBS})
