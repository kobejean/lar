#include <stdint.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "lar/core/utils/json.h"
#include "lar/tracking/tracker.h"
#include "lar/mapping/frame.h"

using namespace std;

std::string getPathPrefix(std::string directory, int id) {
  std::string id_string = std::to_string(id);
  int zero_count = 8 - static_cast<int>(id_string.length());
  std::string prefix = std::string(zero_count, '0') + id_string + '_';
  return directory + prefix;
};

int main(int argc, const char* argv[]){
  string input = "./input/aizu-park-2-proc";
  // string output = "./output/map.g2o";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  std::ifstream map_data_ifs(input + "/map.json");
  std::cout << "parse map" << std::endl;
  nlohmann::json map_data = nlohmann::json::parse(map_data_ifs);
  lar::Map map = map_data;
  lar::Tracker tracker(map);
  
  std::vector<lar::Frame> frames = nlohmann::json::parse(std::ifstream("./input/aizu-park-sunny/frames.json"));
  int successful = 0;
  for (auto& frame : frames) {
    std::cout << std::endl << "LOCALIZING FRAME " << frame.id << std::endl;
    std::string image_path = getPathPrefix("./input/aizu-park-sunny/", frame.id) + "image.jpeg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    Eigen::Matrix3f frameIntrinsics = frame.intrinsics.cast<float>().transpose(); // transpose so that the order of data matches opencv
    cv::Mat intrinsics(3, 3, CV_32FC1, frameIntrinsics.data());
    cv::Mat transform;//(4, 4, CV_64FC1);

    // Extract gravity vector from frame extrinsics    
    Eigen::Vector3d worldGravity(0.0f, -1.0f, 0.0f);
    Eigen::Vector3d cameraGravity = frame.extrinsics.block<3, 3>(0, 0).inverse() * worldGravity;
    // Convert to opencv
    cv::Mat gvec = (cv::Mat_<double>(3,1) << cameraGravity(0), -cameraGravity(1), -cameraGravity(2));

    if (tracker.localize(image, intrinsics, transform, gvec)) {
      std::cout << "transform:" << transform << std::endl;
    }
  }

  std::cout << "Successfully localized " << successful << "/" << frames.size() << " images!" << std::endl;
  return 0;
}
