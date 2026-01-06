# lar_service library (gRPC service implementations)
add_library(lar_service STATIC
  pathfinding_service.cpp
)

target_link_libraries(lar_service PUBLIC
  lar_core
  lar_proto
  gRPC::grpc++
)

target_include_directories(lar_service PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)