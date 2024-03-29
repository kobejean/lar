set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/base64.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/json.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/wgs84.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/anchor.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/landmark.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/landmark_database.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/map.h"
)

add_library(lar_core
  ${HEADER_LIST}
  anchor.cpp
  landmark.cpp
  landmark_database.cpp
  map.cpp
)
target_link_libraries(lar_core 
  g2o::types_slam3d
  opencv_core
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
)

add_subdirectory(data_structures)
add_subdirectory(spacial)