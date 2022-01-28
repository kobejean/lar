set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/bundle_adjustment.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/depth.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/frame_processor.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/map_processor.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/projection.h"
)

add_library(geoar_processing
  bundle_adjustment.cpp
  depth.cpp
  frame_processor.cpp
  map_processor.cpp
  projection.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_processing 
  g2o::types_sba g2o::types_slam3d g2o::solver_eigen
  opencv_core
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
  geoar_mapping
  geoar_tracking
)