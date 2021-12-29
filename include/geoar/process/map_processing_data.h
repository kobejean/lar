#ifndef GEOAR_MAP_PROCESSING_DATA_H
#define GEOAR_MAP_PROCESSING_DATA_H

#include <opencv2/features2d.hpp>

#include <nlohmann/json.hpp>

#include "geoar/core/frame.h"
#include "geoar/core/map.h"
#include "geoar/process/vision.h"

namespace geoar {

  class MapProcessingData {
    public:
      Vision vision;
      Map map;
      std::vector<Frame> frames;
      cv::Mat desc;

      MapProcessingData();
      void loadRawData(std::string directory);
      void loadFrameData(nlohmann::json& frame_data, std::string directory);

    private:
      std::vector<size_t> getLandmarks(Frame &frame, cv::Mat &desc);
      std::map<size_t, size_t> getMatches(cv::Mat &desc);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* GEOAR_MAP_PROCESSING_DATA_H */