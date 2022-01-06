# Create geoar_create_map executable
add_executable(geoar_create_map geoar_create_map.cpp)
target_link_libraries(geoar_create_map
  geoar_process
)
set_target_properties(geoar_create_map
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.geoar.create-map
  # EXCLUDE_FROM_ALL ON
)