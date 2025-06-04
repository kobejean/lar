#include <iostream>
#include <opencv2/calib3d.hpp>

#include "lar/tracking/tracker.h"

namespace lar {

  namespace {

  }

  Tracker::Tracker(Map map) {
    this->map = map;
  }

  bool Tracker::localize(cv::InputArray image, const cv::Mat& intrinsics, cv::Mat &transform) {
    cv::Mat rvec, tvec;
    if (!transform.empty()) {
      fromTransform(transform, rvec, tvec);
    }

    bool success = localize(image, intrinsics, cv::Mat(), rvec, tvec, !transform.empty());
    if (success) {
      toTransform(rvec, tvec, transform);
    }
    return success;
  }

  bool Tracker::localize(cv::InputArray image, const cv::Mat& intrinsics, const cv::Mat& dist_coeffs, cv::Mat &rvec, cv::Mat &tvec, bool use_extrinsic_guess) {
    for (Landmark *landmark : local_landmarks) { landmark->is_matched = false; }
    // get map descriptors
    if (tvec.empty()) {
      local_landmarks.clear();
      for (Landmark *landmark : map.landmarks.all()) {
        local_landmarks.push_back(landmark);
      }
    } else {
      double query_diameter = 25.0;
      Rect query = Rect(Point(tvec.at<double>(0), tvec.at<double>(2)), query_diameter, query_diameter);
      local_landmarks = map.landmarks.find(query);
    }
    cv::Mat map_desc = Landmark::concatDescriptions(local_landmarks);
    // Extract Features
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    std::vector<cv::DMatch> matches = vision.match(desc, map_desc);
    std::cout << "matches.size()" << matches.size() << std::endl;
    if (matches.size() <= 3) return false; 
    
    cv::Mat object_points = objectPoints(local_landmarks, matches);
    cv::Mat image_points = imagePoints(matches, kpts);
    cv::Mat inliers;

    bool success = cv::solvePnPRansac(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec,
      use_extrinsic_guess, 100, 8.0, 0.99, inliers, cv::SolvePnPMethod::SOLVEPNP_ITERATIVE);
    std::cout << "kpts.size()" << kpts.size() << std::endl;
    std::cout << "matches.size()" << matches.size() << std::endl;
    std::cout << "inliers.size()" << inliers.size() << std::endl;

#ifndef LAR_COMPACT_BUILD
    // mark matches
    for (auto &match : matches) {
      local_landmarks[match.trainIdx]->is_matched = true;
    }
#endif

    return success;
  }

  // Private Methods

  cv::Mat Tracker::objectPoints(const std::vector<Landmark*> &landmarks, const std::vector<cv::DMatch>& matches) {
    cv::Mat object_points(matches.size(), 3, CV_32FC1);
    for (size_t i = 0; i < matches.size(); i++) {
      cv::DMatch match = matches[i];
      const Eigen::Vector3d position = landmarks[match.trainIdx]->position;
      object_points.at<float>(i,0) = position.x();
      object_points.at<float>(i,1) = position.y();
      object_points.at<float>(i,2) = position.z();
    }
    return object_points;
  }

  cv::Mat Tracker::imagePoints(const std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts) {
    cv::Mat image_points(matches.size(), 2, CV_32FC1);
    for (size_t i = 0; i < matches.size(); i++) {
      cv::DMatch match = matches[i];
      cv::KeyPoint kpt = kpts[match.queryIdx];
      image_points.at<float>(i,0) = kpt.pt.x;
      image_points.at<float>(i,1) = kpt.pt.y;
    }
    return image_points;
  }

  void Tracker::toTransform(const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &transform) {
    // Convert back to transform
    cv::Mat rtmat(3, 3, CV_64FC1);
    cv::Rodrigues(rvec, rtmat);
    cv::transpose(rtmat,rtmat);
    cv::Mat tvec_inv = -rtmat * tvec;

    transform = (cv::Mat_<double>(4,4) <<
      rtmat.at<double>(0,0), -rtmat.at<double>(0,1),  -rtmat.at<double>(0,2), tvec_inv.at<double>(0),
      rtmat.at<double>(1,0), -rtmat.at<double>(1,1),  -rtmat.at<double>(1,2), tvec_inv.at<double>(1),
      rtmat.at<double>(2,0), -rtmat.at<double>(2,1),  -rtmat.at<double>(2,2), tvec_inv.at<double>(2),
                         0.,                     0.,                      0.,                 1.
    );
  }

  void Tracker::fromTransform(const cv::Mat &transform, cv::Mat &rvec, cv::Mat &tvec) {
    // OpenCV camera transform is the inverse of ARKit (transpose works because it's orthogonal)
    // Also y,z-axis needs to be flipped hence the negative signs
    cv::Mat rmat = (cv::Mat_<double>(3,3) <<
        transform.at<double>(0,0),  transform.at<double>(1,0),  transform.at<double>(2,0),
      -transform.at<double>(0,1), -transform.at<double>(1,1), -transform.at<double>(2,1),
      -transform.at<double>(0,2), -transform.at<double>(1,2), -transform.at<double>(2,2)
    );
    std::cout << "rmat" << rmat << std::endl;
    rvec = cv::Mat(3, 1, CV_64FC1);
    cv::Rodrigues(rmat, rvec);

    // To switch to OpenCV coordinates, the camera transform is inverse of ARKit
    // so calculate inverse translation using: `-R'*t`
    tvec = (cv::Mat_<double>(3,1) << transform.at<double>(0,3), transform.at<double>(1,3), transform.at<double>(2,3));
    tvec = -rmat * tvec;
  }

}