set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/geoar/process/map_processor.h")

add_library(geoar_process map_processor.cpp ${HEADER_LIST})
target_include_directories(geoar_process PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_process 
  ${G2O_LIBS}
  nlohmann_json::nlohmann_json
)