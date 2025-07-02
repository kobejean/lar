set(HEADER_LIST 
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/vision.h"
)

add_library(lar_tracking
  tracker.cpp
  vision.cpp
  ${HEADER_LIST}
)
target_link_libraries(lar_tracking 
  opencv_core opencv_calib3d
  lar_core lar_mapping
)