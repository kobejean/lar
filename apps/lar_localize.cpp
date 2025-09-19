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
  // string localize = "./input/aizu-park-4-proc/";
  string localize = "./input/aizu-park-sunny/";

  std::ifstream map_data_ifs("./output/map/map.json");
  std::cout << "parse map" << std::endl;
  nlohmann::json map_data = nlohmann::json::parse(map_data_ifs);
  lar::Map map = map_data;
  lar::Tracker tracker(map);
  
  std::vector<lar::Frame> frames = nlohmann::json::parse(std::ifstream(localize+"frames.json"));
  int successful = 0;
  for (auto& frame : frames) {
    std::cout << std::endl << "LOCALIZING FRAME " << frame.id << std::endl;
    std::string image_path = getPathPrefix(localize, frame.id) + "image.jpeg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    // Use frame position for spatial query (assuming frame.extrinsics contains camera pose)
    double query_x = frame.extrinsics(0, 3);
    double query_z = frame.extrinsics(2, 3);
    double query_diameter = 20.0; // 20 meter search radius
    
    Eigen::Matrix4d result_transform;
    if (tracker.localize(image, frame, query_x, query_z, query_diameter, result_transform)) {
      frame.extrinsics = result_transform;
      std::cout << "transform:" << frame.extrinsics << std::endl;
      successful++;
    }
  }

  std::cout << "Successfully localized " << successful << "/" << frames.size() << " images!" << std::endl;
  return 0;
}
