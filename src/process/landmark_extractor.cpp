
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "geoar/process/landmark_extractor.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  const float RATIO_TEST_THRESHOLD = 0.2f;
  const float MARGIN_TEST_DISTANCE = 25.f; // TODO: Think of clearer name

  LandmarkExtractor::LandmarkExtractor() {
    detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0008f, 4, 4, cv::KAZE::DIFF_PM_G2);
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
  }

  vector<Landmark> LandmarkExtractor::extractLandmarks(json& frame_data, std::string directory) {
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
    extractFeatures(image, cv::noArray(), kpts, desc);
    cout << "feature count: " << kpts.size() << endl;

    // Match and filter features
    matchAndFilter(kpts, desc);
    cout << "filtered feature count: " << kpts.size() << endl;

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

    return landmarks;
  }

  // Private methods

  void LandmarkExtractor::extractFeatures(cv::InputArray image, cv::InputArray mask, vector<cv::KeyPoint> &kpts, cv::Mat &desc) {

    vector<cv::KeyPoint> new_kpts;
    cv::Mat new_desc;

    detector->detectAndCompute(image, mask, new_kpts, new_desc);

    // Match function uses ratio test and margin test. We can reuse it to filter features 
    // that are not distinct enough by matching the set of descriptions with each other.
    vector<cv::DMatch> matches = match(new_desc, new_desc);

    // Populate `desc` and `kpts` with matched key points and descriptions
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      // Matches should have sucessfully matched with itself, but it is redundant to check if it did
      assert(idx == matches[i].trainIdx);
      kpts.push_back(new_kpts[idx]);
      desc.push_back(new_desc.row(idx));
    }
  }

  void LandmarkExtractor::matchAndFilter(vector<cv::KeyPoint> &kpts, cv::Mat &desc) {

    vector<cv::DMatch> matches = match(desc, all_desc);
    cout << "match count: " << matches.size() << endl;

    // Populate `idx_matched` map
    std::map<int, bool> idx_matched;
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      idx_matched[idx] = true;
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

  vector<Vector3f> LandmarkExtractor::projectKeyPoints(vector<cv::KeyPoint> &kpts, cv::Mat &depth, json& frame_data) {
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

  vector<Landmark> LandmarkExtractor::createLandmarks(vector<Vector3f> &pts3d, vector<cv::KeyPoint> &kpts, cv::Mat &desc) {
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

  void LandmarkExtractor::recordFeatures(vector<Landmark> &landmarks, cv::Mat &desc) {
    // Add new descriptions to `all_desc`
    if (all_desc.rows > 0) {
      cv::vconcat(all_desc, desc, all_desc);
    } else {
      all_desc = desc;
    }
    // Append contents of landmarks to `discoveredLandmarks`
    discoveredLandmarks.reserve(discoveredLandmarks.size() + distance(landmarks.begin(), landmarks.end()));
    discoveredLandmarks.insert(discoveredLandmarks.end(), landmarks.begin(), landmarks.end());
  }

  vector<cv::DMatch> LandmarkExtractor::match(cv::Mat desc1, cv::Mat desc2) {
    vector<cv::DMatch> filtered_matches;
    // We need at least 2 rows to perform ratio test
    if (desc1.rows <= 2 || desc2.rows <= 2) return filtered_matches;
    vector< vector<cv::DMatch> > nn_matches;

    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    // Filter matches
    for (size_t i = 0; i < nn_matches.size(); i++) {
        cv::DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        bool ratio_test = dist1 < RATIO_TEST_THRESHOLD * dist2;
        bool margin_test = dist2 >= MARGIN_TEST_DISTANCE;
        if (ratio_test && margin_test) {
            filtered_matches.push_back(first);
        }
    }

    return filtered_matches;
  }

}