add_subdirectory(core)
if (NOT LAR_COMPACT_BUILD)
  # lar_io (image codec leaf) is only needed by mapping/processing; tracking never links it.
  add_subdirectory(io)
  add_subdirectory(mapping)
  add_subdirectory(processing)
endif()
add_subdirectory(tracking)

if(LAR_BUILD_SERVICE)
  add_subdirectory(service)
endif()