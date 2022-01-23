set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/collection/collection.h"
)

add_library(geoar_collection
  collection.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_collection 
  opencv_core opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
)