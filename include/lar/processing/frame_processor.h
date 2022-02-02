#ifndef LAR_PROCESSING_FRAME_PROCESSOR_H
#define LAR_PROCESSING_FRAME_PROCESSOR_H

#include <opencv2/core/types.hpp>

#include <nlohmann/json.hpp>

#include "lar/mapping/frame.h"
#include "lar/processing/map_processor.h"
#include "lar/tracking/vision.h"

namespace lar {

  class FrameProcessor {
    public:
      Mapper::Data& data;
      Vision vision;

      FrameProcessor(Mapper::Data& data);
      void process(Frame& frame);

    private:
      std::vector<size_t> getLandmarks(Frame& frame, cv::Mat& desc);
      std::map<size_t, size_t> getMatches(cv::Mat& desc);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* LAR_PROCESSING_FRAME_PROCESSOR_H */