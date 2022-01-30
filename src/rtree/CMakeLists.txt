set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/lar/rtree/rtree.h")

add_library(lar_rtree rtree.cpp ${HEADER_LIST})
target_include_directories(lar_rtree PUBLIC ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(lar_rtree 
#   ${G2O_LIBS}
#   nlohmann_json::nlohmann_json
# )