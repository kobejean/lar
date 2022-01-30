set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/lar/processing/bundle_adjustment.h"
  "${PROJECT_SOURCE_DIR}/include/lar/processing/depth.h"
  "${PROJECT_SOURCE_DIR}/include/lar/processing/frame_processor.h"
  "${PROJECT_SOURCE_DIR}/include/lar/processing/map_processor.h"
  "${PROJECT_SOURCE_DIR}/include/lar/processing/projection.h"
)

add_library(lar_processing
  bundle_adjustment.cpp
  depth.cpp
  frame_processor.cpp
  map_processor.cpp
  projection.cpp
  ${HEADER_LIST}
)
target_link_libraries(lar_processing 
  g2o::types_sba g2o::types_slam3d g2o::solver_eigen
  opencv_core
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  lar_core
  lar_mapping
  lar_tracking
)