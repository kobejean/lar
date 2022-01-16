set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/tracking/tracking.h"
)

add_library(geoar_tracking
  tracking.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_tracking 
  g2o::types_sba g2o::types_slam3d g2o::solver_eigen
  opencv_core opencv_features2d opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
)