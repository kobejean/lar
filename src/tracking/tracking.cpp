#include <iostream>
#include <opencv2/calib3d.hpp>

#include "geoar/tracking/tracking.h"

namespace geoar {

  Tracking::Tracking(Map map) {
    this->map = map;
  }

  void Tracking::localize(cv::InputArray image, cv::Mat intrinsics, cv::Mat dist_coeffs, cv::Mat &rvec, cv::Mat &tvec) {
    // Extract Features
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    cv::Mat map_desc = map.landmarks.getDescriptions();
    std::vector<cv::DMatch> matches = vision.match(desc, map_desc);
    cv::Mat object_points = objectPoints(matches);
    cv::Mat image_points = imagePoints(matches, kpts);
    cv::Mat inliers;

    // std::cout << "object_points" << object_points << std::endl;
    // std::cout << "image_points" << image_points << std::endl;

    cv::solvePnPRansac(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec,
      false, 100, 8.0, 0.99, inliers, cv::SolvePnPMethod::SOLVEPNP_ITERATIVE);
    std::cout << "kpts.size()" << kpts.size() << std::endl;
    std::cout << "matches.size()" << matches.size() << std::endl;
    std::cout << "inliers.size()" << inliers.size() << std::endl;
  }

  // Private Methods

  cv::Mat Tracking::objectPoints(std::vector<cv::DMatch> matches) {
    cv::Mat object_points(matches.size(), 3, CV_32FC1);
    for (size_t i = 0; i < matches.size(); i++) {
      cv::DMatch match = matches[i];
      Eigen::Vector3d position = map.landmarks[match.trainIdx].position;
      object_points.at<float>(i,0) = position.x();
      object_points.at<float>(i,1) = position.y();
      object_points.at<float>(i,2) = position.z();
    }
    return object_points;
  }

  cv::Mat Tracking::imagePoints(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kpts) {
    cv::Mat image_points(matches.size(), 2, CV_32FC1);
    for (size_t i = 0; i < matches.size(); i++) {
      cv::DMatch match = matches[i];
      cv::KeyPoint kpt = kpts[match.queryIdx];
      image_points.at<float>(i,0) = kpt.pt.x;
      image_points.at<float>(i,1) = kpt.pt.y;
    }
    return image_points;
  }

}