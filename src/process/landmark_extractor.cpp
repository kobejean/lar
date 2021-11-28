
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

    // Create filename prefix
    string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    string prefix = string(zero_count, '0') + id_string + '_';

    string img_filepath = directory + '/' + prefix + "image.jpeg";
    cout << "loading: " << img_filepath << endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // string confidence_filepath = directory + '/' + prefix + "confidence.pfm";
    // cout << "loading: " << confidence_filepath << endl;
    // cv::Mat confidence = cv::imread(confidence_filepath, cv::IMREAD_UNCHANGED);
    // cv::resize(confidence, confidence, image.size(), 0, 0, cv::INTER_NEAREST);

    vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    extractFeatures(image, cv::noArray(), kpts, desc);
    cout << "feature count: " << kpts.size() << endl;


    string depth_filepath = directory + '/' + prefix + "depth.pfm";
    cout << "loading: " << depth_filepath << endl;
    cv::Mat depth = cv::imread(depth_filepath, cv::IMREAD_UNCHANGED);
    cv::resize(depth, depth, image.size(), 0, 0, cv::INTER_LINEAR);

    vector<Landmark> empty;
    return empty;
  }

  // Private methods

  void LandmarkExtractor::extractFeatures(cv::InputArray image, cv::InputArray mask, vector<cv::KeyPoint> &kpts, cv::Mat &desc) {

    vector<cv::KeyPoint> all_kpts;
    cv::Mat all_desc;

    detector->detectAndCompute(image, mask, all_kpts, all_desc);

    // match with self to make sure we have distinct features
    vector<cv::DMatch> matches = match(all_desc, all_desc);

    vector<Landmark> landmarks;
    for(size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      desc.push_back(all_desc.row(idx));
      kpts.push_back(all_kpts[idx]);
    }

    // Landmark::concatDescriptions(landmarks, desc);

    // discoveredLandmarks.reserve(discoveredLandmarks.size() + distance(landmarks.begin(),landmarks.end()));
    // discoveredLandmarks.insert(discoveredLandmarks.end(),landmarks.begin(),landmarks.end());
  }

  vector<cv::DMatch> LandmarkExtractor::match(cv::Mat desc1, cv::Mat desc2) {
    vector<vector<cv::DMatch>> nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<cv::DMatch> filtered_matches;

    for(size_t i = 0; i < nn_matches.size(); i++) {
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