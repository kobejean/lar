# Create lar_create_map executable
add_executable(lar_create_map lar_create_map.cpp)
target_link_libraries(lar_create_map
  lar_processing
)
set_target_properties(lar_create_map
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.create-map
  # EXCLUDE_FROM_ALL ON
)

# Create lar_create_map executable
add_executable(lar_localize lar_localize.cpp)
target_link_libraries(lar_localize
  lar_tracking
  opencv_core opencv_features2d opencv_imgcodecs
)
set_target_properties(lar_localize
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.localize
  # EXCLUDE_FROM_ALL ON
)