set(HEADER_LIST
"${PROJECT_SOURCE_DIR}/include/geoar/core/frame.h"
"${PROJECT_SOURCE_DIR}/include/geoar/core/landmark.h"
"${PROJECT_SOURCE_DIR}/include/geoar/core/map.h"
)

add_library(geoar_core frame.cpp landmark.cpp map.cpp ${HEADER_LIST})
target_include_directories(geoar_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(geoar_core 
  ${G2O_LIBS}
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
)