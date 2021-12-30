#ifndef GEOAR_MAP_PROCESSOR_H
#define GEOAR_MAP_PROCESSOR_H

#include "geoar/process/bundle_adjustment.h"
#include "geoar/process/map_processing_data.h"

namespace geoar {

  class MapProcessor {
    public:
      MapProcessingData data;
      BundleAdjustment bundle_adjustment;

      MapProcessor();
      void createMap(std::string directory);
      void loadData(std::string directory);
  };
}

#endif /* GEOAR_MAP_PROCESSOR_H */