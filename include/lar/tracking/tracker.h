#ifndef LAR_TRACKING_TRACKER_H
#define LAR_TRACKING_TRACKER_H

#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>

#include "lar/core/map.h"
#include "lar/mapping/frame.h"
#include "lar/tracking/vision.h"

namespace lar {

  class Tracker {
    public:
      Vision vision;
      Map map;
      std::vector<Landmark*> local_landmarks;
      std::vector<std::pair<Landmark*, cv::KeyPoint>> inliers;

      Tracker(Map map);
      bool localize(cv::InputArray image, const Frame &frame, Eigen::Matrix4d &extrinsics);
      bool localize(cv::InputArray image, const cv::Mat& intrinsics, cv::Mat &transform, const cv::Mat &gvec);
      bool localize(cv::InputArray image, const cv::Mat& intrinsics, const cv::Mat& dist_coeffs, cv::Mat &rvec, cv::Mat &tvec, const cv::Mat &gvec);

    private:
      cv::UsacParams usac_params;
      cv::Mat objectPoints(const std::vector<Landmark*> &landmarks, const std::vector<cv::DMatch>& matches);
      cv::Mat imagePoints(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts);
      void toTransform(const cv::Mat& rvec, const cv::Mat& tvec, cv::Mat& transform);
      void fromTransform(const cv::Mat& transform, cv::Mat& rvec, cv::Mat& tvec);
  };

}

#endif /* LAR_TRACKING_TRACKER_H */
