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
# add_dependencies(geoar_core OpenCV)
target_include_directories(geoar_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_core 
  types_sba types_slam3d solver_eigen
  opencv_core
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
)