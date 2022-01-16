set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/process/bundle_adjustment.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/depth.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/frame_processing.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processing_data.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processing.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/projection.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/vision.h"
)

add_library(geoar_process
  bundle_adjustment.cpp
  depth.cpp
  frame_processing.cpp
  map_processing_data.cpp
  map_processing.cpp
  projection.cpp
  vision.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_process 
  g2o::types_sba g2o::types_slam3d g2o::solver_eigen
  opencv_core opencv_features2d opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
)