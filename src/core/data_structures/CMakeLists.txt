set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/core/data_structures/unordered_array.h"
)


target_sources(lar_core PRIVATE 
  ${HEADER_LIST}
)