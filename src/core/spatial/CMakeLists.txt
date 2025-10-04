set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/spatial/point.h"
  "${PROJECT_SOURCE_DIR}/include/lar/core/spatial/rect.h"
)


target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
  point.cpp
  rect.cpp
)

add_subdirectory(region_tree)