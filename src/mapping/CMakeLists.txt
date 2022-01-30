set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/lar/mapping/frame.h"
  "${PROJECT_SOURCE_DIR}/include/lar/mapping/location_matcher.h"
  "${PROJECT_SOURCE_DIR}/include/lar/mapping/mapper.h"
)

add_library(lar_mapping
  frame.cpp
  location_matcher.cpp
  mapper.cpp
  ${HEADER_LIST}
)
target_link_libraries(lar_mapping 
  opencv_core opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  lar_core
)