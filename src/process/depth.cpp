#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "geoar/process/depth.h"

namespace geoar {
  Depth::Depth(cv::Size img_size) {
    _img_size = img_size;
  }
  
  std::vector<double> Depth::at(std::vector<cv::KeyPoint> const &kpts) {
    assert(_map.size() == _img_size);

    std::vector<double> values;
    values.reserve(kpts.size());

    for (cv::KeyPoint const& kpt : kpts) {
      int x = std::round(kpt.pt.x), y = std::round(kpt.pt.y);
      double value = _map.at<double>(y, x);
      values.push_back(value);
    }

    return values;
  }

  void Depth::unloadDepthMap() {
    _map = cv::Mat();
  }


  // SavedDepth

  SavedDepth::SavedDepth(cv::Size img_size, std::string filepath): Depth(img_size) {
    _filepath = filepath;
  }

  void SavedDepth::loadDepthMap() {
    std::cout << "loading: " << _filepath << std::endl;
    _map = cv::imread(_filepath, cv::IMREAD_UNCHANGED);
    cv::resize(_map, _map, _img_size, 0, 0, cv::INTER_LINEAR);
  }
}