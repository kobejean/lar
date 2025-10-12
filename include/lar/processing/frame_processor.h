#ifndef LAR_PROCESSING_FRAME_PROCESSOR_H
#define LAR_PROCESSING_FRAME_PROCESSOR_H

#include <opencv2/core/types.hpp>

#include <nlohmann/json.hpp>

#include "lar/mapping/frame.h"
#include "lar/mapping/mapper.h"
#include "lar/tracking/vision.h"

namespace lar {

  class FrameProcessor {
    public:
      std::shared_ptr<Mapper::Data> data;
      Vision vision;
      std::vector<Landmark*> local_landmarks;

      FrameProcessor(std::shared_ptr<Mapper::Data> data);
      void process(Frame& frame);

    private:
      void extractLandmarks(const Frame &frame, const cv::Mat &desc, const std::vector<cv::KeyPoint>& kpts, const std::vector<float>& depth, std::vector<Landmark*> &results);
      std::map<size_t, size_t> getMatches(const cv::Mat &query_desc, const cv::Mat &train_desc, const std::vector<cv::KeyPoint>& kpts);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* LAR_PROCESSING_FRAME_PROCESSOR_H */