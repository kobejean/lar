add_subdirectory(g2o_ba_demo)

# Create geoar_bundle_adjustment executable
add_executable(geoar_bundle_adjustment geoar_bundle_adjustment.cpp)
target_link_libraries(geoar_bundle_adjustment
  geoar_process
)

# Create geoar_create_map executable
add_executable(geoar_create_map geoar_create_map.cpp)
target_link_libraries(geoar_create_map
  geoar_process
)