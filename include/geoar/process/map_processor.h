#ifndef GEOAR_MAP_PROCESSOR_H
#define GEOAR_MAP_PROCESSOR_H

#include "geoar/process/graph_construction.h"
#include "geoar/process/map_processing_data.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class MapProcessor {
    public:
      MapProcessingData data;
      GraphConstruction graph_construction;

      MapProcessor();
      void createMap(std::string directory);
  };
}

#endif /* GEOAR_MAP_PROCESSOR_H */