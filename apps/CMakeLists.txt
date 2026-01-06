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

# Create lar_localize executable
add_executable(lar_localize lar_localize.cpp)
target_link_libraries(lar_localize
  lar_tracking lar_mapping
  opencv_core opencv_features2d opencv_imgcodecs
)
set_target_properties(lar_localize
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.localize
  # EXCLUDE_FROM_ALL ON
)

# Create lar_refine_colmap executable
add_executable(lar_refine_colmap lar_refine_colmap.cpp)
target_link_libraries(lar_refine_colmap
  lar_processing
)
set_target_properties(lar_refine_colmap
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.refine-colmap
  # EXCLUDE_FROM_ALL ON
)

# Create lar_rtree executable
add_executable(lar_rtree lar_rtree.cpp)
target_link_libraries(lar_rtree
  lar_core
)
set_target_properties(lar_rtree
  PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.rtree
  # EXCLUDE_FROM_ALL ON
)

if(LAR_BUILD_SERVER)
  add_executable(lar_server lar_server.cpp)
  target_link_libraries(lar_server
    lar_service
    lar_core
    gRPC::grpc++_reflection
  )
  set_target_properties(lar_server
    PROPERTIES
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER com.local-ar.server
  )
endif()