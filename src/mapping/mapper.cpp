#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "lar/core/utils/json.h"
#include "lar/mapping/mapper.h"

namespace fs = std::filesystem;

namespace lar {

  Mapper::Mapper(fs::path directory) {
    fs::create_directory(directory);
    this->data.directory = directory;
  }

  void Mapper::addFrame(Frame frame, cv::InputArray image, cv::InputArray depth, cv::InputArray confidence) {
    frame.id = static_cast<int>(data.frames.size());
    std::string path_prefix = data.getPathPrefix(frame.id).string();
    cv::imwrite(path_prefix + "image.jpeg", image);
    cv::imwrite(path_prefix + "depth.pfm", depth);
    cv::imwrite(path_prefix + "confidence.pfm", confidence);
    data.frames.push_back(frame);
  }


  void Mapper::addPosition(Eigen::Vector3d position, long long timestamp) {
    location_matcher.recordPosition(timestamp, position);
    data.gps_obs = location_matcher.matches;
  }

  void Mapper::addLocation(Eigen::Vector3d location, Eigen::Vector3d accuracy, long long timestamp) {
    location_matcher.recordLocation(timestamp, location, accuracy);
    data.gps_obs = location_matcher.matches;
  }

  void Mapper::writeMetadata() {
    {
    std::ofstream file(data.directory / "frames.json");
    nlohmann::json json = data.frames;
    file << json << std::endl;
    }
    {
    std::ofstream file(data.directory / "gps.json");
    nlohmann::json json = data.gps_obs;
    file << json << std::endl;
    }
  }

  void Mapper::readMetadata() {
    {
    std::ifstream ifs(data.directory / "frames.json");
    data.frames = nlohmann::json::parse(ifs);
    }
    {
    std::ifstream ifs(data.directory / "gps.json");
    data.gps_obs = nlohmann::json::parse(ifs);
    }
  }

}