#ifndef GEOAR_MAP_PROCESSING_H
#define GEOAR_MAP_PROCESSING_H

#include <string>
#include "geoar/process/map_processing_data.h"

namespace geoar {

  class MapProcessing {
    public:
      MapProcessing();
      void createMap(std::string in_dir, std::string out_dir);
      MapProcessingData loadData(std::string directory);
  };
}

#endif /* GEOAR_MAP_PROCESSING_H */