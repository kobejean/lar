#include <stdint.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "lar/core/utils/json.h"
#include "lar/tracking/tracker.h"
#include "lar/tracking/detectors/sift.h"

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

  cv::Mat image1 = cv::imread(input + "/00000004_image.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat intrinsics(3, 3, CV_32FC1);
  intrinsics.at<float>(0,0) = 1594.2728271484375;
  intrinsics.at<float>(1,1) = 1594.2728271484375;
  intrinsics.at<float>(0,2) = 952.7379150390625;
  intrinsics.at<float>(1,2) = 714.167236328125;
  intrinsics.at<float>(2,2) = 1.;
  cv::Mat transform;//(4, 4, CV_64FC1);
  // transform.at<double>(0,3) = 0.;
  // transform.at<double>(1,3) = 0.;
  // transform.at<double>(2,3) = 0.;
  // transform.at<double>(0,0) = 1.;
  // transform.at<double>(1,1) = 1.;
  // transform.at<double>(2,2) = 1.;
  // transform.at<double>(3,3) = 1.;

  lar::Tracker tracker(map);
  std::cout << "parse map" << std::endl;
  tracker.localize(image1, intrinsics, transform);
  std::cout << "transform:" << transform << std::endl;


  lar::SIFT sift;
  std::vector<cv::KeyPoint> kpts1;
  cv::Mat desc1;
  std::vector<cv::KeyPoint> kpts2;
  cv::Mat desc2;

  sift.detect(image1, kpts1);
  std::string output_kpts1 = "./output/sift/kpts1.jpeg";
  cv::Mat image_kpts1;
  cv::drawKeypoints(image1, kpts1, image_kpts1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imwrite(output_kpts1, image_kpts1);


  cv::Mat image2 = cv::imread(input + "/00000003_image.jpeg", cv::IMREAD_GRAYSCALE);
  sift.detect(image2, kpts2);
  std::string output_kpts2 = "./output/sift/kpts2.jpeg";
  cv::Mat image_kpts2;
  cv::drawKeypoints(image2, kpts2, image_kpts2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imwrite(output_kpts2, image_kpts2);

  std::cout << "kpts1: " << kpts1.size() << std::endl;
  std::cout << "kpts2: " << kpts2.size() << std::endl;
  return 0;
}
