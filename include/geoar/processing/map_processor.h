#ifndef GEOAR_PROCESSING_MAP_PROCESSOR_H
#define GEOAR_PROCESSING_MAP_PROCESSOR_H

#include <filesystem>
#include <string>
#include <opencv2/features2d.hpp>

#include "geoar/mapping/mapper.h"
#include "geoar/core/map.h"

namespace fs = std::filesystem;

namespace geoar {

  class MapProcessor {
    public:
      Mapper::Data* data;
      
      MapProcessor(Mapper::Data& data);
      void createMap(std::string out_dir);
  };
}

#endif /* GEOAR_PROCESSING_MAP_PROCESSOR_H */