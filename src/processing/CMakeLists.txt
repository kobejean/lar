set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/bundle_adjustment.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/depth.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/frame_processing.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/map_processing_data.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/map_processing.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/processing/projection.h"
)

add_library(geoar_processing
  bundle_adjustment.cpp
  depth.cpp
  frame_processing.cpp
  map_processing_data.cpp
  map_processing.cpp
  projection.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_processing 
  g2o::types_sba g2o::types_slam3d g2o::solver_eigen
  opencv_core opencv_features2d opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
  geoar_tracking
)