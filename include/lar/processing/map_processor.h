#ifndef LAR_PROCESSING_MAP_PROCESSOR_H
#define LAR_PROCESSING_MAP_PROCESSOR_H

#include <string>

#include "lar/mapping/mapper.h"
#include "lar/core/map.h"

namespace lar {

  class MapProcessor {
    public:
      Mapper::Data& data;
      
      MapProcessor(Mapper::Data& data);
      void process();
      void createMap(std::string out_dir);
  };
}

#endif /* LAR_PROCESSING_MAP_PROCESSOR_H */