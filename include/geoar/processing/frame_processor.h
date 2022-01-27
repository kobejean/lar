#ifndef GEOAR_PROCESSING_FRAME_PROCESSOR_H
#define GEOAR_PROCESSING_FRAME_PROCESSOR_H

#include <opencv2/features2d.hpp>

#include <nlohmann/json.hpp>

#include "geoar/mapping/frame.h"
#include "geoar/processing/map_processor.h"
#include "geoar/tracking/vision.h"

namespace geoar {

  class FrameProcessor {
    public:
      Mapper::Data* data;
      Vision vision;

      FrameProcessor(Mapper::Data &data);
      void process(Frame& frame);

    private:
      std::vector<size_t> getLandmarks(Frame &frame, cv::Mat &desc);
      std::map<size_t, size_t> getMatches(cv::Mat &desc);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* GEOAR_PROCESSING_FRAME_PROCESSOR_H */