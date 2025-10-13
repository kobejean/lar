#include <iostream>
#include <opencv2/calib3d.hpp>

#include "lar/tracking/tracker.h"

namespace lar {

  Tracker::Tracker(Map& map, cv::Size imageSize) : vision(imageSize), map(map), last_gravity_angle_difference(0.0) {
    usac_params = cv::UsacParams();
    usac_params.confidence=0.99;
    usac_params.isParallel=false;
    usac_params.loIterations=10;
    usac_params.loMethod=cv::LocalOptimMethod::LOCAL_OPTIM_SIGMA;
    usac_params.loSampleSize=50;
    usac_params.maxIterations=5000;
    usac_params.sampler=cv::SamplingMethod::SAMPLING_PROSAC;
    usac_params.score=cv::ScoreMethod::SCORE_METHOD_MAGSAC;
    // usac_params.final_polisher = cv::PolishingMethod::POLISHING_MAGSAC;
    // usac_params.final_polisher_iterations = 10;
    usac_params.threshold=8.0;
  }

  bool Tracker::localize(cv::InputArray image, const Frame &frame, double query_x, double query_z, double query_diameter, Eigen::Matrix4d &result_transform, const Eigen::Matrix4d &initial_guess, bool use_initial_guess) {
    Eigen::Matrix3f frameIntrinsics = frame.intrinsics.cast<float>().transpose(); // transpose so that the order of data matches opencv
    cv::Mat intrinsics(3, 3, CV_32FC1, frameIntrinsics.data());

    // Extract gravity vector from frame extrinsics    
    Eigen::Vector3d worldGravity(0.0f, -1.0f, 0.0f);
    Eigen::Vector3d cameraGravity = frame.extrinsics.block<3, 3>(0, 0).inverse() * worldGravity;
    Eigen::Vector3d cameraPosition = frame.extrinsics.block<3, 1>(0, 3);

    // Convert to opencv
    cv::Mat gvec = (cv::Mat_<double>(3,1) << cameraGravity(0), -cameraGravity(1), -cameraGravity(2));

    // Prepare initial guess if provided
    cv::Mat rvec, tvec;
    if (use_initial_guess) {
      cv::Mat initial_transform(4, 4, CV_64FC1);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          initial_transform.at<double>(i, j) = initial_guess(i, j);
        }
      }
      fromTransform(initial_transform, rvec, tvec);
    }

    bool success = localize(image, intrinsics, cv::Mat(), rvec, tvec, gvec, query_x, query_z, query_diameter);
    
    if (success) {
      // Convert result to transform matrix
      cv::Mat transform(4, 4, CV_64FC1);
      toTransform(rvec, tvec, transform);
      
      // Copy to result_transform
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          result_transform(i, j) = transform.at<double>(i, j);
        }
      }
      
      // Add observations for inliers
      for (auto& pair : inliers) {
        Landmark& landmark = *pair.first;
        cv::KeyPoint kpt = pair.second;
        Eigen::Vector3d landmarkWorldPos = landmark.position;
        Eigen::Vector3d distance_vec = landmarkWorldPos - cameraPosition;
        float depth = (float)distance_vec.norm();

        Landmark::Observation obs{
          .frame_id=frame.id,
          .timestamp=frame.timestamp,
          .cam_pose=frame.extrinsics,
          .kpt=kpt,
          .depth=depth,
          .depth_confidence=0,
          .surface_normal=landmark.orientation,
        };
        landmark.obs.push_back(obs);
      }
    }
    
    return success;
  }

  bool Tracker::localize(cv::InputArray image, const cv::Mat& intrinsics, const cv::Mat& dist_coeffs, cv::Mat &rvec, cv::Mat &tvec, const cv::Mat &gvec, double query_x, double query_z, double query_diameter) {
    for (Landmark *landmark : local_landmarks) { landmark->is_matched = false; }
    this->matches.clear();
    this->inliers.clear();
    
    // get map descriptors using explicit spatial query parameters
    if (query_diameter <= 0.0) {
      // No spatial filtering - use all landmarks
      local_landmarks.clear();
      for (Landmark *landmark : map.landmarks.all()) {
        local_landmarks.push_back(landmark);
      }
    } else {
      // Use spatial query with explicit parameters
      std::cout << "Spatial query: Point(" << query_x << ", " << query_z << ") diameter=" << query_diameter << std::endl;
      Rect query = Rect(Point(query_x, query_z), query_diameter, query_diameter);
      local_landmarks.clear();
      map.landmarks.find(query, local_landmarks);
    }
    
    // Limit local_landmarks to prevent OpenCV crashes (max 250k landmarks)
    constexpr size_t MAX_LANDMARKS = (1 << 18)-1; // 262,143
    if (local_landmarks.size() > MAX_LANDMARKS) {
      std::cout << "Warning: Limiting local_landmarks from " << local_landmarks.size() 
                << " to " << MAX_LANDMARKS << " to prevent OpenCV crash" << std::endl;
      local_landmarks.resize(MAX_LANDMARKS);
    }
    
    std::cout << "local_landmarks.size(): " << local_landmarks.size() << std::endl;
    cv::Mat map_desc = Landmark::concatDescriptions(local_landmarks);
    
    // Extract Features
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    
    std::vector<cv::DMatch> matches = vision.match(desc, map_desc, kpts);
    if (matches.size() <= 3) return false; 
    
    cv::Mat object_points = objectPoints(local_landmarks, matches);
    cv::Mat image_points = imagePoints(matches, kpts);
    cv::Mat inliers;

    bool success = cv::solvePnPRansac(object_points, image_points, intrinsics, dist_coeffs, rvec, tvec, inliers, usac_params);
    
    std::cout << "kpts.size(): " << kpts.size() << std::endl;
    std::cout << "matches.size(): " << matches.size() << std::endl;
    std::cout << "inliers.size(): " << inliers.size() << std::endl;

    if (!gvec.empty() && !this->checkGravityVector(rvec, gvec, 3.0f)) return false;

// #ifndef LAR_COMPACT_BUILD
    // Store all feature matches
    for (const auto& match : matches) {
      Landmark* landmark = local_landmarks[match.trainIdx];
      this->matches.emplace_back(landmark, kpts[match.queryIdx]);
    }
    
    // Store inliers (subset of matches)
    for (int i = 0; i < inliers.rows; i++) {
      int match_idx = inliers.at<int>(i);
      auto& match = matches[match_idx];
      Landmark* landmark = local_landmarks[match.trainIdx];
      landmark->is_matched = true;
      this->inliers.emplace_back(landmark, kpts[match.queryIdx]);
    }
    
// #endif

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
                         0.,                     0.,                      0.,                     1.
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

  // Private method to check gravity vector consistency and store angle difference
  bool Tracker::checkGravityVector(const cv::Mat& rvec, const cv::Mat& gvec, float angleTolerance) {
    // World gravity vector (pointing down in Y direction)
    cv::Mat worldGravity = (cv::Mat_<double>(3,1) << 0.0, -1.0, 0.0);
    
    // Convert rvec to rotation matrix
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    
    // Transform world gravity to camera coordinates using estimated rotation
    // Since we're going from world to camera, we use the rotation matrix directly
    cv::Mat estimatedCameraGravity = rmat * worldGravity;
    
    // Convert gvec to double if needed for consistency
    cv::Mat measuredGravity;
    gvec.convertTo(measuredGravity, CV_64F);
    
    // Normalize both vectors (they should already be normalized, but just in case)
    cv::normalize(estimatedCameraGravity, estimatedCameraGravity);
    cv::normalize(measuredGravity, measuredGravity);
    
    // Calculate dot product
    double dotProduct = estimatedCameraGravity.dot(measuredGravity);
    
    // Clamp dot product to valid range for acos
    dotProduct = std::max(-1.0, std::min(1.0, dotProduct));
    
    // Calculate angle
    double angleRadians = std::acos(dotProduct);
    double angleDegrees = angleRadians * 180.0 / M_PI;
    
    // Store the angle difference
    last_gravity_angle_difference = angleDegrees;
    
    std::cout << "Gravity vector angle difference: " << angleDegrees << " degrees" << std::endl;
    
    return angleDegrees <= angleTolerance;
  }

  // Getter for the last calculated gravity angle difference
  double Tracker::getLastGravityAngleDifference() const {
    return last_gravity_angle_difference;
  }

}