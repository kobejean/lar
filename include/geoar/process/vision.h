#ifndef VISION_H
#define VISION_H

#include <opencv2/features2d.hpp>

using namespace std;

namespace geoar {

  class Vision {
    public:
      cv::Ptr<cv::AKAZE> detector;
      cv::BFMatcher matcher;

      Vision();

      void extractFeatures(cv::InputArray image, cv::InputArray mask, vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      vector<cv::DMatch> match(cv::Mat &desc1, cv::Mat &desc2);
  };

}

#endif /* VISION_H */