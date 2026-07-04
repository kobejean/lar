# lar_io: the ONLY target that links an image codec (stb, JPEG-only + a small PFM codec).
# Kept as a leaf so that core/tracking never transitively pull in codec code, letting
# tracking-only apps dead-strip all image I/O out of the final binary.
add_library(lar_io
  image_io.cpp
  "${PROJECT_SOURCE_DIR}/include/lar/io/image_io.h"
)

target_link_libraries(lar_io
  PUBLIC
    opencv_core
)

# Vendored stb headers (single-file, header-only, zero external deps -> same on all platforms).
target_include_directories(lar_io
  PRIVATE
    "${PROJECT_SOURCE_DIR}/thirdparty/stb"
)
