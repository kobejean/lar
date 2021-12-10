set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/process/graph_construction.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processing_data.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processor.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/process/vision.h"
)

add_library(geoar_process
  map_processing_data.cpp
  graph_construction.cpp
  map_processor.cpp
  vision.cpp
  ${HEADER_LIST}
)
target_include_directories(geoar_process PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_process 
  ${G2O_LIBS}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  # Eigen3::Eigen
  geoar_core
)