#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "geoar/core/utils/json.h"
#include "geoar/collection/collection.h"

namespace fs = std::filesystem;

namespace geoar {

  Collection::Collection(fs::path directory) {
    this->directory = directory;
    fs::create_directory(directory);
  }

  void Collection::addFrame(cv::InputArray image, cv::InputArray depth, cv::InputArray confidence, FrameMetadata metadata) {
    int id = static_cast<int>(frames.size());
    metadata.id = id;
    std::string path_prefix = getPathPrefix(id).string();
    cv::imwrite(path_prefix + "image.jpeg", image);
    cv::imwrite(path_prefix + "depth.pfm", depth);
    cv::imwrite(path_prefix + "confidence.pfm", confidence);
    frames.push_back(metadata);

    std::ofstream file(directory / "frames.json");
    nlohmann::json json = frames;
    file << json << std::endl;
  }

  void Collection::addGPSObservation(GPSObservation observation) {
    gps_observations.push_back(observation);

    std::ofstream file(directory / "gps_observations.json");
    nlohmann::json json = gps_observations;
    file << json << std::endl;
  }

  fs::path Collection::getPathPrefix(int id) {
    std::string id_string = std::to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    return directory / prefix;
  }
}