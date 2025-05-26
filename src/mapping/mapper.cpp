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
    this->data->directory = directory;
  }

  void Mapper::addFrame(Frame frame, cv::InputArray image, cv::InputArray depth, cv::InputArray confidence) {
    frame.id = static_cast<int>(data->frames.size());
    std::string path_prefix = data->getPathPrefix(frame.id).string();
    cv::imwrite(path_prefix + "image.jpeg", image);
    cv::imwrite(path_prefix + "depth.pfm", depth);
    cv::imwrite(path_prefix + "confidence.pfm", confidence);
    data->frames.push_back(frame);
  }

  void Mapper::addPosition(Eigen::Vector3d position, long long timestamp) {
    location_matcher.recordPosition(timestamp, position);
    data->gps_obs = location_matcher.matches;
  }

  void Mapper::addLocation(Eigen::Vector3d location, Eigen::Vector3d accuracy, long long timestamp) {
    location_matcher.recordLocation(timestamp, location, accuracy);
    data->gps_obs = location_matcher.matches;
  }

  Anchor& Mapper::createAnchor(Transform &transform) {
    Anchor& anchor = data->map.createAnchor(transform);
#ifndef LAR_COMPACT_BUILD
    if (data->frames.size() > 0) {
      anchor.frame_id = data->frames.back().id;
      anchor.relative_transform = data->frames.back().extrinsics.inverse() * transform.matrix();
    }
#endif
    return anchor;
  }

  void Mapper::writeMetadata() {
    nlohmann::json frames_json = data->frames;
    std::ofstream(data->directory / "frames.json") << frames_json << std::endl;
    
    nlohmann::json gps_json = data->gps_obs;
    std::ofstream(data->directory / "gps.json") << gps_json << std::endl;
  }

  void Mapper::readMetadata() {
    data->frames = nlohmann::json::parse(std::ifstream(data->directory / "frames.json"));
    data->gps_obs = nlohmann::json::parse(std::ifstream(data->directory / "gps.json"));
  }

}