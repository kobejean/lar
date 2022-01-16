#include <iostream>
#include <opencv2/calib3d.hpp>

#include "geoar/tracking/tracking.h"

namespace geoar {

  Tracking::Tracking(Map map) {
    this->map = map;
  }

  void Tracking::localize(cv::InputArray image, cv::Mat intrinsics, cv::Mat &transform) {
    cv::Mat rvec, tvec;
    if (!transform.empty()) {
      fromTransform(transform, rvec, tvec);
    }

    localize(image, intrinsics, cv::Mat(), rvec, tvec, !transform.empty());
    toTransform(rvec, tvec, transform);
  }

  void Tracking::localize(cv::InputArray image, cv::Mat intrinsics, cv::Mat dist_coeffs, cv::Mat &rvec, cv::Mat &tvec, bool use_extrinsic_guess) {
    // Extract Features
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    cv::Mat map_desc = map.landmarks.getDescriptions();
    std::vector<cv::DMatch> matches = vision.match(desc, map_desc);
    cv::Mat object_points = objectPoints(matches);
    cv::Mat image_points = imagePoints(matches, kpts);
    cv::Mat inliers;

    cv::solvePnPRansac(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec,
      use_extrinsic_guess, 100, 8.0, 0.99, inliers, cv::SolvePnPMethod::SOLVEPNP_ITERATIVE);
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

  void Tracking::toTransform(const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &transform) {
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

  void Tracking::fromTransform(const cv::Mat &transform, cv::Mat &rvec, cv::Mat &tvec) {
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