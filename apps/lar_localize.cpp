#include <stdint.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "lar/core/utils/json.h"
#include "lar/tracking/tracker.h"

using namespace std;

int main(int argc, const char* argv[]){
  string input = "./input/snapshot";
  // string output = "./output/map.g2o";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  std::ifstream map_data_ifs(input + "/map.json");
  nlohmann::json map_data = nlohmann::json::parse(map_data_ifs);
  lar::Map map = map_data;

  cv::Mat image = cv::imread(input + "/00000001_image.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat intrinsics(3, 3, CV_32FC1);
  intrinsics.at<float>(0,0) = 1594.2728271484375;
  intrinsics.at<float>(1,1) = 1594.2728271484375;
  intrinsics.at<float>(0,2) = 952.7379150390625;
  intrinsics.at<float>(1,2) = 714.167236328125;
  intrinsics.at<float>(2,2) = 1.;
  cv::Mat transform;

  lar::Tracker tracker(map);
  tracker.localize(image, intrinsics, transform);
  std::cout << "transform:" << transform << std::endl;
  return 0;
}
