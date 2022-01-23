#ifndef GEOAR_PROCESSING_MAP_PROCESSING_DATA_H
#define GEOAR_PROCESSING_MAP_PROCESSING_DATA_H

#include <opencv2/features2d.hpp>

#include "geoar/core/frame.h"
#include "geoar/core/map.h"

namespace geoar {

  class MapProcessingData {
    public:
      Map map;
      std::vector<Frame> frames;
      cv::Mat desc;

      MapProcessingData();
  };

}

#endif /* GEOAR_PROCESSING_MAP_PROCESSING_DATA_H */