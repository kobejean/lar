#include <iostream>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "geoar/core/landmark.h"
#include "geoar/process/depth.h"
#include "geoar/process/projection.h"
#include "geoar/process/frame_processing.h"

namespace geoar {

  FrameProcessing::FrameProcessing(MapProcessingData &data) {
    this->data = &data;
  }

  Frame FrameProcessing::process(nlohmann::json& frame_data, std::string directory) {
    int id = frame_data["id"];
    Frame frame(frame_data);

    // Create filename paths
    std::string path_prefix = getPathPrefix(id, directory);
    std::string img_filepath = path_prefix + "image.jpeg";

    // Load image
    std::cout << "loading: " << img_filepath << std::endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // Extract features
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), frame.kpts, desc);
    std::cout << "features: " << frame.kpts.size() << std::endl;

    // Retreive depth values
    SavedDepth depth(image.size(), path_prefix);
    frame.depth = depth.depthAt(frame.kpts);
    frame.confidence = depth.confidenceAt(frame.kpts);

    // Get landmarks
    frame.landmarks = getLandmarks(frame, desc);

    return frame;
  }

  // Private methods

  std::vector<size_t> FrameProcessing::getLandmarks(Frame &frame, cv::Mat &desc) {
    // Filter out features that have been matched
    std::map<size_t, size_t> matches = getMatches(desc);
    Projection projection(frame.frame_data);

    size_t landmark_count = frame.kpts.size();
    std::vector<Landmark> new_landmarks;
    new_landmarks.reserve(landmark_count);
    std::vector<size_t> landmark_ids;
    landmark_ids.reserve(landmark_count);
    
    size_t new_landmark_id = data->map.landmarks.size();
    for (size_t i = 0; i < landmark_count; i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark
        Eigen::Vector3d pt3d = projection.projectToWorld(frame.kpts[i].pt, frame.depth[i]);
        Landmark landmark(pt3d, desc.row(i), new_landmark_id);
        landmark.recordSighting(frame.transform);

        landmark_ids.push_back(new_landmark_id);
        new_landmarks.push_back(landmark);
        new_landmark_id++;
      } else {
        // We have a match so just push the match index
        landmark_ids.push_back(matches[i]);
        data->map.landmarks[matches[i]].recordSighting(frame.transform);
      }
    }

    data->map.landmarks.insert(new_landmarks);
    return landmark_ids;
  }

  std::map<size_t, size_t> FrameProcessing::getMatches(cv::Mat &desc) {
    // Get matches
    vector<cv::DMatch> matches = vision.match(desc, data->desc);
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

    // Add new descriptions to `data->desc`
    if (data->desc.rows > 0) {
      cv::vconcat(data->desc, unmatched_desc, data->desc);
    } else {
      data->desc = unmatched_desc;
    }

    return idx_matched;
  }

  std::string FrameProcessing::getPathPrefix(int id, std::string directory) {
    std::string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    return directory + '/' + prefix;
  }

}