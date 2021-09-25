set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/geoar/rtree/rtree.h")

add_library(geoar_rtree rtree.cpp ${HEADER_LIST})
target_include_directories(geoar_rtree PUBLIC ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(geoar_rtree 
#   ${G2O_LIBS}
#   nlohmann_json::nlohmann_json
# )