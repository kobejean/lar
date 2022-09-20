set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/spacial/region_tree.h"
  node.h
)

target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
  region_tree.cpp
  node.cpp
  node_partition.cpp
)
