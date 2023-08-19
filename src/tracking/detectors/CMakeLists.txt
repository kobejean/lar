set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/detectors/gaussian.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/detectors/sift.h"
)


target_sources(lar_tracking PRIVATE 
  ${HEADER_LIST}
  sift.cpp
)
