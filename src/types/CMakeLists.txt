set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/geoar/types/frame.h")

add_library(geoar_types frame.cpp ${HEADER_LIST})
target_include_directories(geoar_types PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_types 
  ${G2O_LIBS}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
)