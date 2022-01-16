#ifndef GEOAR_FRAME_PROCESSING_H
#define GEOAR_FRAME_PROCESSING_H

#include <opencv2/features2d.hpp>

#include <nlohmann/json.hpp>

#include "geoar/core/frame.h"
#include "geoar/process/map_processing_data.h"
#include "geoar/tracking/vision.h"

namespace geoar {

  class FrameProcessing {
    public:
      MapProcessingData* data;
      Vision vision;

      FrameProcessing(MapProcessingData &data);
      Frame process(nlohmann::json& frame_data, std::string directory);

    private:
      std::vector<size_t> getLandmarks(Frame &frame, cv::Mat &desc);
      std::map<size_t, size_t> getMatches(cv::Mat &desc);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* GEOAR_FRAME_PROCESSING_H */