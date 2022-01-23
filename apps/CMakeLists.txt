# Create geoar_create_map executable
add_executable(geoar_create_map geoar_create_map.cpp)
target_link_libraries(geoar_create_map
  geoar_processing
)
set_target_properties(geoar_create_map
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.geoar.create-map
  # EXCLUDE_FROM_ALL ON
)

# Create geoar_create_map executable
add_executable(geoar_localize geoar_localize.cpp)
target_link_libraries(geoar_localize
  geoar_tracking
  opencv_core opencv_features2d opencv_imgcodecs
)
set_target_properties(geoar_localize
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.geoar.localize
  # EXCLUDE_FROM_ALL ON
)