cmake_minimum_required(VERSION 3.0)
project(GeoARCore)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
set(CMAKE_CXX_STANDARD 14)


find_package(Eigen3 3.3 REQUIRED)
if (TARGET Eigen3::Eigen)
  set(G2O_EIGEN3_EIGEN_TARGET Eigen3::Eigen)
else()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif ()


FIND_PACKAGE(G2O REQUIRED)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  # ${CSPARSE_INCLUDE_DIR}
  # ${Cholmod_INCLUDE_DIR}
  ${G2O_INCLUDE_DIR}
)
link_directories(
  ${G2O_LIB_DIR}
)

# add_library(${PROJECT_NAME} SHARED
# )
LIST(APPEND G2O_LIBS
# cxsparse
# cholmod
# g2o_cli g2o_simulator
# g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
# g2o_core g2o_solver_csparse g2o_solver_structure_only
# g2o_types_sba g2o_types_slam3d g2o_csparse_extension
# g2o_solver_dense g2o_stuff
# g2o_types_sclam2d g2o_parser g2o_solver_pcg
# g2o_types_data g2o_types_sim3

g2o_core g2o_types_sba g2o_types_slam3d g2o_solver_dense
g2o_solver_eigen

# g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
# g2o_core g2o_types_sba g2o_solver_dense
# g2o_solver_eigen
)


add_executable(ba_demo src/demo/ba_demo.cpp)
target_link_libraries(ba_demo 
${G2O_LIBS}
)
if(CHOLMOD_FOUND)
  target_link_libraries(ba_demo g2o_solver_cholmod)
else()
  target_link_libraries(ba_demo g2o_solver_eigen)
endif()
target_compile_definitions(ba_demo PUBLIC "G2O_USE_VENDORED_CERES")
