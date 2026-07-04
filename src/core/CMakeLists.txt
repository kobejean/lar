set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/base64.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/json.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/wgs84.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/utils/transform.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/anchor.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/landmark.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/landmark_database.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/map.h"
  # Frame is a fundamental pose/intrinsics type; it lives in core so that tracking can
  # use it without depending on lar_mapping (which pulls in image I/O). Header path is
  # still include/lar/mapping/frame.h for source compatibility.
  "${PROJECT_SOURCE_DIR}/include/lar/mapping/frame.h"
)

add_library(lar_core
  ${HEADER_LIST}
  anchor.cpp
  frame.cpp
  landmark.cpp
  landmark_database.cpp
  map.cpp
  utils/transform.cpp
)
target_link_libraries(lar_core 
  g2o::types_slam3d
  opencv_core
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
)

add_subdirectory(data_structures)
add_subdirectory(spatial)