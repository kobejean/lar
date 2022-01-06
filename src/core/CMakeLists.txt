set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/geoar/core/frame.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/core/landmark.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/core/landmark_database.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/core/map.h"
)

add_library(geoar_core
  ${HEADER_LIST}
  frame.cpp
  landmark.cpp
  landmark_database.cpp
  map.cpp
)
target_include_directories(geoar_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_core 
  g2o::types_slam3d
  opencv_core
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
)