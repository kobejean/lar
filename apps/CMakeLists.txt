add_subdirectory(g2o_ba_demo)

# Create bundle_adjustment executable
add_executable(bundle_adjustment bundle_adjustment.cpp)
target_link_libraries(bundle_adjustment
${G2O_LIBS}
nlohmann_json::nlohmann_json
)
message(G2O_LIBS ${G2O_LIBS})