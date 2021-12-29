set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/process/bundle_adjustment.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processing_data.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processor.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/projection.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/vision.h"
)

add_library(geoar_process
  map_processing_data.cpp
  bundle_adjustment.cpp
  map_processor.cpp
  projection.cpp
  vision.cpp
  ${HEADER_LIST}
)
target_include_directories(geoar_process PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_process 
  types_sba types_slam3d solver_eigen
  opencv_core opencv_features2d opencv_imgcodecs
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  geoar_core
)

# Create G2O_LIBS list
# list(APPEND G2O_LIBS types_sba types_slam3d solver_dense)
# list(APPEND G2O_LIBS solver_eigen)