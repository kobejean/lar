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

    // Extract features
    cv::Mat desc;
    std::vector<cv::KeyPoint> kpts;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    std::cout << "features: " << kpts.size() << std::endl;

    // Retreive depth values
    SavedDepth depth(image.size(), frame.intrinsics, frame.extrinsics, path_prefix);
    auto depth_values = depth.depthAt(kpts);
    auto confidence_values = depth.confidenceAt(kpts);
    auto surface_normals = depth.surfaceNormaslAt(kpts);
    auto landmark_ids = getLandmarkIds(frame, desc, kpts, depth_values);

    // Create landmark observations
    for (size_t i=0; i<kpts.size(); i++) {
      Landmark::Observation obs{
        .landmark_id=landmark_ids[i],
        .frame_id=frame.id,
        .timestamp=frame.timestamp,
        .cam_position=frame.extrinsics.block<3,1>(0,3),
        .kpt=kpts[i],
        .depth=depth_values[i],
        .depth_confidence=confidence_values[i],
        .surface_normal=surface_normals[i],
      };
      frame.obs.push_back(obs);
      Landmark& landmark = data->map.landmarks[landmark_ids[i]];
      landmark.recordObservation(obs);
    }

    frame.processed = true;
  }

  // Private methods

  std::vector<size_t> FrameProcessor::getLandmarkIds(const Frame &frame, const cv::Mat &desc, const std::vector<cv::KeyPoint>& kpts, const std::vector<float>& depth) {
    // Filter out features that have been matched
    std::map<size_t, size_t> matches = getMatches(desc);
    Projection projection(frame.intrinsics, frame.extrinsics);

    size_t landmark_count = kpts.size();
    std::vector<Landmark> new_landmarks;
    new_landmarks.reserve(landmark_count);
    std::vector<size_t> landmark_ids;
    landmark_ids.reserve(landmark_count);
    
    size_t new_landmark_id = data->map.landmarks.size();
    for (size_t i = 0; i < landmark_count; i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark
        Eigen::Vector3d pt3d = projection.projectToWorld(kpts[i].pt, depth[i]);
        Landmark landmark(pt3d, desc.row(i), new_landmark_id);

        landmark_ids.push_back(new_landmark_id);
        new_landmarks.push_back(landmark);
        new_landmark_id++;
      } else {
        // We have a match so just push the match index
        landmark_ids.push_back(matches[i]);
      }
    }

    data->map.landmarks.insert(new_landmarks);
    return landmark_ids;
  }

  std::map<size_t, size_t> FrameProcessor::getMatches(const cv::Mat &desc) {
    // Get matches
    std::vector<cv::DMatch> matches = vision.match(desc, data->map.landmarks.getDescriptions());
    std::cout << "matches: " << matches.size() << std::endl;

    // Populate `idx_matched` map
    std::map<size_t, size_t> idx_matched;
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      idx_matched[idx] = matches[i].trainIdx;
    }

    // Populate unmatched descriptions
    cv::Mat unmatched_desc; // TODO: see if there's a way to reserve capacity
    for (size_t i = 0; i < (unsigned)desc.rows; i++) {
      if (idx_matched.find(i) == idx_matched.end()) {
        unmatched_desc.push_back(desc.row(i));
      }
    }

    return idx_matched;
  }

}