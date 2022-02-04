#include <iostream>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lar/core/landmark.h"
#include "lar/processing/depth.h"
#include "lar/processing/projection.h"
#include "lar/processing/frame_processor.h"

namespace lar {

  FrameProcessor::FrameProcessor(Mapper::Data& data) : data(data) {
  }

  void FrameProcessor::process(Frame& frame) {
    if (frame.processed) return;

    // Create filename paths
    std::string path_prefix = data.getPathPrefix(frame.id).string();
    std::string img_filepath = path_prefix + "image.jpeg";

    // Load image
    std::cout << "loading: " << img_filepath << std::endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // Extract features
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), frame.kpts, desc);
    std::cout << "features: " << frame.kpts.size() << std::endl;

    // Retreive depth values
    SavedDepth depth(image.size(), frame.intrinsics, frame.extrinsics, path_prefix);
    frame.depth = depth.depthAt(frame.kpts);
    frame.confidence = depth.confidenceAt(frame.kpts);
    frame.surface_normals = depth.surfaceNormaslAt(frame.kpts);

    // Get landmarks
    frame.landmark_ids = getLandmarks(frame, desc);
    frame.processed = true;
  }

  // Private methods

  std::vector<size_t> FrameProcessor::getLandmarks(Frame &frame, cv::Mat &desc) {
    // Filter out features that have been matched
    std::map<size_t, size_t> matches = getMatches(desc);
    Projection projection(frame.intrinsics, frame.extrinsics);

    size_t landmark_count = frame.kpts.size();
    std::vector<Landmark> new_landmarks;
    new_landmarks.reserve(landmark_count);
    std::vector<size_t> landmark_ids;
    landmark_ids.reserve(landmark_count);
    
    size_t new_landmark_id = data.map.landmarks.size();
    for (size_t i = 0; i < landmark_count; i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark
        Eigen::Vector3d pt3d = projection.projectToWorld(frame.kpts[i].pt, frame.depth[i]);
        Landmark landmark(pt3d, desc.row(i), new_landmark_id);
        landmark.recordObservation({
          .timestamp=frame.timestamp,
          .cam_position=(frame.extrinsics.block<3,1>(0,3)),
          .surface_normal=frame.surface_normals[i]
        });

        landmark_ids.push_back(new_landmark_id);
        new_landmarks.push_back(landmark);
        new_landmark_id++;
      } else {
        // We have a match so just push the match index
        landmark_ids.push_back(matches[i]);
        data.map.landmarks[matches[i]].recordObservation({
          .timestamp=frame.timestamp,
          .cam_position=(frame.extrinsics.block<3,1>(0,3)),
          .surface_normal=frame.surface_normals[i]
        });
      }
    }

    data.map.landmarks.insert(new_landmarks);
    return landmark_ids;
  }

  std::map<size_t, size_t> FrameProcessor::getMatches(cv::Mat &desc) {
    // Get matches
    std::vector<cv::DMatch> matches = vision.match(desc, data.map.landmarks.getDescriptions());
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