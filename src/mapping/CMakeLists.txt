set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/mapping/location_matcher.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/mapping/mapper.h"
)

add_library(geoar_mapping
  location_matcher.cpp
  mapper.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_mapping 
  opencv_core opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
)