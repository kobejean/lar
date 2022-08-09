set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/spacial/point.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/spacial/rectangle.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/spacial/region_tree.h"
)


target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
  point.cpp
  rectangle.cpp
  region_tree.cpp
)