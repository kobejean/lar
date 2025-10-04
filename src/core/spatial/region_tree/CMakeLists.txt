set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/spatial/region_tree.h"
  node.h
  node_traits.h
  node_partition.h
)

target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
  region_tree.cpp
  node.cpp
  node_partition.cpp
)
