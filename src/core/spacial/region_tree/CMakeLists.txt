set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/spacial/region_tree.h"
  _node.h
)

target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
  region_tree.cpp
  _node.cpp
  _node_partition.cpp
)
