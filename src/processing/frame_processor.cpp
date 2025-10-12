#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lar/core/landmark.h"
#include "lar/processing/depth.h"
#include "lar/processing/projection.h"
#include "lar/processing/frame_processor.h"

namespace lar {

  FrameProcessor::FrameProcessor(std::shared_ptr<Mapper::Data> data) : data(data) {
  }

  void FrameProcessor::process(Frame& frame) {
    if (frame.processed) return;

    // Create filename paths
    std::string path_prefix = data->getPathPrefix(frame.id).string();
    std::string img_filepath = path_prefix + "image.jpeg";

    // Load image
    std::cout << "loading: " << img_filepath << std::endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
    std::cout << image.size() << std::endl;

    // Extract features
    cv::Mat desc;
    std::vector<cv::KeyPoint> kpts;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    std::cout << "features: " << kpts.size() << std::endl;
    std::cout << "total features: " << data->map.landmarks.size() << std::endl;

    // Retreive depth values
    SavedDepth depth(image.size(), frame.intrinsics, frame.extrinsics, path_prefix);
    auto depth_values = depth.depthAt(kpts);
    auto confidence_values = depth.confidenceAt(kpts);
    auto surface_normals = depth.surfaceNormaslAt(kpts);
    local_landmarks.clear();
    extractLandmarks(frame, desc, kpts, depth_values, local_landmarks);

    // Create record observations
    for (size_t i=0; i<kpts.size(); i++) {
      Landmark::Observation obs{
        .frame_id=frame.id,
        .timestamp=frame.timestamp,
        .cam_pose=frame.extrinsics,
        .kpt=kpts[i],
        .depth=depth_values[i],
        .depth_confidence=confidence_values[i],
        .surface_normal=surface_normals[i],
      };
      data->map.landmarks.addObservation(local_landmarks[i]->id, obs);
    }

    frame.processed = true;
  }

  // Private methods

  // TODO: See if there is a better way to deal with the side effect of inserting into map.landmarks
  void FrameProcessor::extractLandmarks(const Frame &frame, const cv::Mat &desc, const std::vector<cv::KeyPoint>& kpts, const std::vector<float>& depth, std::vector<Landmark*> &results) {
    // get local descriptors
    static const double QUERY_DIAMETER = 30.0;
    static const int LOCAL_LANDMARKS_LIMIT = (1 << 17);
    Rect query_rect = Rect(Point(frame.extrinsics(0,3), frame.extrinsics(2,3)), QUERY_DIAMETER, QUERY_DIAMETER);
    data->map.landmarks.find(query_rect, results, LOCAL_LANDMARKS_LIMIT);
    const cv::Mat &local_desc = Landmark::concatDescriptions(results);
    std::cout << "local_landmarks: " << results.size() << std::endl;

    // get matches
    std::map<size_t, size_t> matches = getMatches(desc, local_desc, kpts);
    
    size_t landmark_count = kpts.size();
    std::vector<Landmark> new_landmarks;
    std::vector<size_t> landmark_ids;
    landmark_ids.reserve(landmark_count);
    Projection projection(frame.intrinsics, frame.extrinsics);
    
    for (size_t i = 0; i < landmark_count; i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark with ID 0 (will be assigned by insert)
        Eigen::Vector3d pt3d = projection.projectToWorld(kpts[i].pt, depth[i]);
        new_landmarks.emplace_back(pt3d, desc.row(i), 0);
      }
    }
    data->map.landmarks.insert(new_landmarks, &results);
  }

  std::map<size_t, size_t> FrameProcessor::getMatches(const cv::Mat &query_desc, const cv::Mat &train_desc, const std::vector<cv::KeyPoint> &kpts) {
    // Get matches
    std::vector<cv::DMatch> matches = vision.match(query_desc, train_desc, kpts);
    std::cout << "matches: " << matches.size() << std::endl;

    // Populate `idx_matched` map
    std::map<size_t, size_t> idx_matched;
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      idx_matched[idx] = local_landmarks[matches[i].trainIdx]->id;
    }

    // Populate unmatched descriptions
    cv::Mat unmatched_desc; // TODO: see if there's a way to reserve capacity
    for (size_t i = 0; i < (unsigned)query_desc.rows; i++) {
      if (idx_matched.find(i) == idx_matched.end()) {
        unmatched_desc.push_back(query_desc.row(i));
      }
    }

    return idx_matched;
  }

}