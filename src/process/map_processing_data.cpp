
#include "geoar/process/map_processing_data.h"

namespace geoar {

  MapProcessingData::MapProcessingData() {
  }

  void MapProcessingData::loadRawData(string directory) {
    ifstream metadata_ifs(directory + "/metadata.json");
    json metadata = json::parse(metadata_ifs);

    for (json frame_data : metadata["frames"]) {
      loadFrameData(frame_data, directory);
    }
  }

  void MapProcessingData::loadFrameData(json& frame_data, std::string directory) {
    int id = frame_data["id"];
    Frame frame(frame_data);

    // Create filename paths
    std::string path_prefix = getPathPrefix(id, directory);
    std::string img_filepath = path_prefix + "image.jpeg";
    std::string depth_filepath = path_prefix + "depth.pfm";

    // Load image
    cout << "loading: " << img_filepath << endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // Extract features
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), frame.kpts, desc);
    cout << "features: " << frame.kpts.size() << endl;

    frame.depth = getDepthValues(frame.kpts, depth_filepath, image.size());
    frame.landmarks = getLandmarks(frame.kpts, desc, frame.depth, frame_data);

    this->frames.push_back(frame);
  }

  // Private methods

  vector<size_t> MapProcessingData::getLandmarks(vector<cv::KeyPoint> &kpts, cv::Mat &desc, vector<float> &depth, json& frame_data) {
    // Filter out features that have been matched
    std::map<size_t, size_t> matches = getMatches(desc);
    Projection projection(frame_data);

    vector<size_t> landmarks;
    landmarks.reserve(kpts.size());

    for (size_t i = 0; i < kpts.size(); i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark
        Vector3d pt3d = projection.projectToWorld(kpts[i].pt, depth[i]);
        Landmark landmark(pt3d, kpts[i], desc.row(i));

        landmarks.push_back(map.landmarkDatabase.landmarks.size());
        map.landmarkDatabase.landmarks.push_back(landmark);
      } else {
        // We have a match so just push the match index
        landmarks.push_back(matches[i]);
        map.landmarkDatabase.landmarks[matches[i]].sightings++;
      }
    }

    return landmarks;
  }

  vector<float> MapProcessingData::getDepthValues(vector<cv::KeyPoint> &kpts, std::string depth_filepath, cv::Size img_size) {
    // Load depth map
    cout << "loading: " << depth_filepath << endl;
    cv::Mat depth = cv::imread(depth_filepath, cv::IMREAD_UNCHANGED);
    cv::resize(depth, depth, img_size, 0, 0, cv::INTER_LINEAR);

    vector<float> depth_values;
    depth_values.reserve(kpts.size());

    for (cv::KeyPoint const& kpt : kpts) {
      // Get depth value
      int idx_x = round(kpt.pt.x), idx_y = round(kpt.pt.y);
      float depth_value = depth.at<float>(idx_y, idx_x);
      depth_values.push_back(depth_value);
    }

    return depth_values;
  }

  std::map<size_t, size_t> MapProcessingData::getMatches(cv::Mat &desc) {
    // Get matches
    vector<cv::DMatch> matches = vision.match(desc, this->desc);
    cout << "matches: " << matches.size() << endl;

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

    // Add new descriptions to `this->desc`
    if (this->desc.rows > 0) {
      cv::vconcat(this->desc, unmatched_desc, this->desc);
    } else {
      this->desc = unmatched_desc;
    }

    return idx_matched;
  }

  std::string MapProcessingData::getPathPrefix(int id, std::string directory) {
    std::string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    return directory + '/' + prefix;
  }

}