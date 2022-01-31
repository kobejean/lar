#ifndef LAR_TRACKING_TRACKER_H
#define LAR_TRACKING_TRACKER_H

#include <opencv2/core/types.hpp>

#include "lar/core/map.h"
#include "lar/tracking/vision.h"

namespace lar {

  class Tracker {
    public:
      Vision vision;
      Map map;

      Tracker(Map map);
      bool localize(cv::InputArray image, cv::Mat intrinsics, cv::Mat &transform);
      bool localize(cv::InputArray image, cv::Mat intrinsics, cv::Mat dist_coeffs, cv::Mat &rvec, cv::Mat &tvec, bool use_extrinsic_guess);

    private:
      cv::Mat objectPoints(std::vector<cv::DMatch> matches);
      cv::Mat imagePoints(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kpts);
      void toTransform(const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &transform);
      void fromTransform(const cv::Mat &transform, cv::Mat &rvec, cv::Mat &tvec);
  };

}

#endif /* LAR_TRACKING_TRACKER_H */