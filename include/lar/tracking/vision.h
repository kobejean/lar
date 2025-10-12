#ifndef LAR_TRACKING_VISION_H
#define LAR_TRACKING_VISION_H

#include "sift/sift.h"
#include <opencv2/features2d.hpp>

namespace lar {

  class Vision {
    public:
      // cv::Ptr<cv::Feature2D> detector;
      cv::Ptr<SIFT> detector;
      cv::FlannBasedMatcher flann_matcher;
      cv::BFMatcher bf_matcher;

      Vision();

      void extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      std::vector<cv::DMatch> match(const cv::Mat &desc1, const cv::Mat &desc2,
                                     const std::vector<cv::KeyPoint>& kpts);

    private:
      std::vector<cv::DMatch> matchOneWay(const cv::Mat& desc1, const cv::Mat& desc2,
                                           const std::vector<cv::KeyPoint>& kpts);
  };

}

#endif /* LAR_TRACKING_VISION_H */