#ifndef GEOAR_VISION_H
#define GEOAR_VISION_H

#include <opencv2/features2d.hpp>

namespace geoar {

  class Vision {
    public:
      cv::Ptr<cv::AKAZE> detector;
      cv::BFMatcher matcher;

      Vision();

      void extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      std::vector<cv::DMatch> match(cv::Mat &desc1, cv::Mat &desc2);
  };

}

#endif /* GEOAR_VISION_H */