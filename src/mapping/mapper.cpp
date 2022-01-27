#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "geoar/core/utils/json.h"
#include "geoar/mapping/mapper.h"

namespace fs = std::filesystem;

namespace geoar {

  Mapper::Mapper(fs::path directory) {
    this->directory = directory;
    fs::create_directory(directory);
  }

  void Mapper::addFrame(cv::InputArray image, cv::InputArray depth, cv::InputArray confidence, FrameMetadata metadata) {
    metadata.id = static_cast<int>(frames.size());
    std::string path_prefix = getPathPrefix(metadata.id).string();
    cv::imwrite(path_prefix + "image.jpeg", image);
    cv::imwrite(path_prefix + "depth.pfm", depth);
    cv::imwrite(path_prefix + "confidence.pfm", confidence);
    frames.push_back(metadata);
  }


  void Mapper::addPosition(Eigen::Vector3d position, long long timestamp) {
    location_matcher.recordPosition(timestamp, position);
    gps_observations = location_matcher.matches;
  }

  void Mapper::addLocation(Eigen::Vector3d location, Eigen::Vector3d accuracy, long long timestamp) {
    location_matcher.recordLocation(timestamp, location, accuracy);
    gps_observations = location_matcher.matches;
  }

  fs::path Mapper::getPathPrefix(int id) {
    std::string id_string = std::to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    return directory / prefix;
  }

  void Mapper::writeMetadata() {
    {
    std::ofstream file(directory / "frames.json");
    nlohmann::json json = frames;
    file << json << std::endl;
    }
    {
    std::ofstream file(directory / "gps_observations.json");
    nlohmann::json json = gps_observations;
    file << json << std::endl;
    }
  }
}