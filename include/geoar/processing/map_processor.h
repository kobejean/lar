#ifndef GEOAR_PROCESSING_MAP_PROCESSOR_H
#define GEOAR_PROCESSING_MAP_PROCESSOR_H

#include <string>

#include "geoar/mapping/mapper.h"
#include "geoar/core/map.h"

namespace geoar {

  class MapProcessor {
    public:
      Mapper::Data* data;
      
      MapProcessor(Mapper::Data& data);
      void process();
      void createMap(std::string out_dir);
  };
}

#endif /* GEOAR_PROCESSING_MAP_PROCESSOR_H */