#ifndef GEOAR_PROCESSING_MAP_PROCESSING_H
#define GEOAR_PROCESSING_MAP_PROCESSING_H

#include <string>
#include "geoar/processing/map_processing_data.h"

namespace geoar {

  class MapProcessing {
    public:
      MapProcessing();
      void createMap(std::string in_dir, std::string out_dir);
      MapProcessingData loadData(std::string directory);
  };
}

#endif /* GEOAR_PROCESSING_MAP_PROCESSING_H */