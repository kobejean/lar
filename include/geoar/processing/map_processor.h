#ifndef GEOAR_PROCESSING_MAP_PROCESSOR_H
#define GEOAR_PROCESSING_MAP_PROCESSOR_H

#include <string>
#include <opencv2/features2d.hpp>

#include "geoar/core/frame.h"
#include "geoar/core/map.h"

namespace geoar {

  class MapProcessor {
    public:
      class Data {
        public:
          Map map;
          std::vector<Frame> frames;
          cv::Mat desc;
          Data() {};
      };

      MapProcessor();
      void createMap(std::string in_dir, std::string out_dir);
      Data loadData(std::string directory);
  };
}

#endif /* GEOAR_PROCESSING_MAP_PROCESSOR_H */