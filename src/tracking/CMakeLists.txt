set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/geoar/tracking/tracker.h"
  "${PROJECT_SOURCE_DIR}/include/geoar/tracking/vision.h"
)

add_library(geoar_tracking
  tracker.cpp
  vision.cpp
  ${HEADER_LIST}
)
target_link_libraries(geoar_tracking 
  opencv_core opencv_calib3d
  geoar_core
)