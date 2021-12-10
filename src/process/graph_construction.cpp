#include <stdint.h>

#include <iostream>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "geoar/process/graph_construction.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  GraphConstruction::GraphConstruction(MapProcessingData &data) {
    this->data = &data;
  }

  void GraphConstruction::processRawData(string directory) {
    ifstream metadata_ifs(directory + "/metadata.json");
    json metadata = json::parse(metadata_ifs);

    for (json frame_data : metadata["frames"]) {
      processFrameData(frame_data, directory);
    }
  }

  void GraphConstruction::processFrameData(json& frame_data, std::string directory) {
    int id = frame_data["id"];

    // Create filename paths
    std::string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    std::string img_filepath = directory + '/' + prefix + "image.jpeg";
    std::string depth_filepath = directory + '/' + prefix + "depth.pfm";

    // Load image
    cout << "loading: " << img_filepath << endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // Extract features
    vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), kpts, desc);
    cout << "features: " << kpts.size() << endl;

    // Match and filter features
    matchAndFilter(kpts, desc);
    cout << "filtered features: " << kpts.size() << endl;

    // Load depth map
    cout << "loading: " << depth_filepath << endl;
    cv::Mat depth = cv::imread(depth_filepath, cv::IMREAD_UNCHANGED);
    cv::resize(depth, depth, image.size(), 0, 0, cv::INTER_LINEAR);

    // Get projected points
    vector<Vector3f> pts3d = projectKeyPoints(kpts, depth, frame_data);

    // Create landmarks by combining data
    vector<Landmark> landmarks = createLandmarks(pts3d, kpts, desc);

    // Add feature history
    recordFeatures(landmarks, desc);
  }

  // Private methods

  void GraphConstruction::matchAndFilter(vector<cv::KeyPoint> &kpts, cv::Mat &desc) {

    vector<cv::DMatch> matches = vision.match(desc, data->desc);
    cout << "matches: " << matches.size() << endl;

    // Populate `idx_matched` map
    std::map<int, bool> idx_matched;
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      idx_matched[idx] = true;
      data->map.landmarkDatabase.landmarks[idx].sightings++;
    }

    // Populate unmatched features
    vector<cv::KeyPoint> unmatched_kpts;
    cv::Mat unmatched_desc;
    assert(kpts.size() == desc.rows);
    for (size_t i = 0; i < kpts.size(); i++) {
      if (idx_matched.find(i) == idx_matched.end()) {
        unmatched_kpts.push_back(kpts[i]);
        unmatched_desc.push_back(desc.row(i));
      }
    }

    // Replace `kpts` with `unmatched_kpts` and `desc` with `unmatched_desc` in-place
    kpts.clear();
    kpts.reserve(kpts.size() + distance(unmatched_kpts.begin(), unmatched_kpts.end()));
    kpts.insert(kpts.end(), unmatched_kpts.begin(), unmatched_kpts.end());
    unmatched_desc.copyTo(desc);
  }

  vector<Vector3f> GraphConstruction::projectKeyPoints(vector<cv::KeyPoint> &kpts, cv::Mat &depth, json& frame_data) {
    // Parse camera transform
    json t = frame_data["transform"];
    Matrix3f rotation;
    rotation << t[0][0], t[1][0], t[2][0],
                t[0][1], t[1][1], t[2][1],
                t[0][2], t[1][2], t[2][2]; 
    Vector3f translation(t[3][0], t[3][1], t[3][2]);

    // Parse intrinsics properties
    json intrinsics = frame_data["intrinsics"];
    json principle_point = intrinsics["principlePoint"];
    float focal_length = intrinsics["focalLength"];
    float principle_point_x = principle_point["x"];
    float principle_point_y = principle_point["y"];

    // Project key points
    vector<Vector3f> pts3d;
    for (cv::KeyPoint const& kpt : kpts) {
      // Get depth value
      int idx_x = round(kpt.pt.x), idx_y = round(kpt.pt.y);
      float depth_value = depth.at<float>(idx_y, idx_x);

      // Use intrinsics and depth to project image coordinates to 3d camera space point
      float cam_x = (kpt.pt.x - principle_point_x) * depth_value / focal_length;
      float cam_y = -(kpt.pt.y - principle_point_y) * depth_value / focal_length;
      float cam_z = -depth_value;
      Vector3f cam_point(cam_x, cam_y, cam_z);

      // Convert camera space point to world space point
      Vector3f pt = rotation * cam_point + translation;
      pts3d.push_back(pt);
    }

    return pts3d;
  }

  vector<Landmark> GraphConstruction::createLandmarks(vector<Vector3f> &pts3d, vector<cv::KeyPoint> &kpts, cv::Mat &desc) {
    assert(pts3d.size() == kpts.size() && kpts.size() == desc.rows);
    size_t count = pts3d.size();

    // Create and populate vector of landmarks
    vector<Landmark> landmarks;
    for (size_t i = 0; i < count; i++) {
      Landmark landmark(pts3d[i], kpts[i], desc.row(i));
      landmarks.push_back(landmark);
    }

    return landmarks;
  }

  void GraphConstruction::recordFeatures(vector<Landmark> &landmarks, cv::Mat &desc) {
    // Add new descriptions to `data->desc`
    if (data->desc.rows > 0) {
      cv::vconcat(data->desc, desc, data->desc);
    } else {
      data->desc = desc;
    }
    // Append contents of landmarks to `map->landmarks`
    data->map.landmarkDatabase.addLandmarks(landmarks);
  }

}