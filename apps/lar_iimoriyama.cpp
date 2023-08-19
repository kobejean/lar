#include <stdint.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "lar/core/utils/json.h"
#include "lar/mapping/frame.h"
#include "lar/tracking/tracker.h"

using namespace std;

Eigen::Vector2d imageCoordinatesFrom(Eigen::Matrix3d intrinsics, Eigen::Matrix4d transform, Eigen::Vector3d point) {
  Eigen::Vector4d position4;
  position4 << point, 1.;
  position4 = transform.inverse() * position4;

  Eigen::Vector3d position3 = position4.head<3>();
  position3(1) = -position3(1);
  position3(2) = -position3(2);
  position3 = intrinsics * position3;
  position3 /= position3(2);
  return position3.head<2>();
}

std::vector<cv::KeyPoint> getKeyPoints(Eigen::Matrix3d intrinsics, Eigen::Matrix4d transform, std::vector<lar::Landmark> landmarks) {
  std::vector<cv::KeyPoint> kpts;
  for (auto landmark : landmarks) {
    Eigen::Vector2d coords = imageCoordinatesFrom(intrinsics, transform, landmark.position);
    cv::KeyPoint kpt;
    kpt.pt.x = coords(0);
    kpt.pt.y = coords(1);
    kpts.push_back(kpt);
  }
  return kpts;
}

Eigen::Vector3d projectPoint(lar::Frame frame, int x, int y, double depth) {
  Eigen::Vector3d point3;
  point3 << x, y, 1.;
  point3 *= depth;
  point3 = frame.intrinsics.inverse() * point3;
  point3(1) = -point3(1);
  point3(2) = -point3(2);
  Eigen::Vector4d point4;
  point4 << point3, 1.;
  point4 = frame.extrinsics * point4;
  return point4.head<3>();
}

int main(int argc, const char* argv[]){
  string input = "./input/iimori1";
  string input2 = "./input/iimori3";
  string output = "./output/iimoriyama";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  std::vector<lar::Frame> frames = nlohmann::json::parse(std::ifstream(input + "/frames.json"));
  Eigen::Vector3d point = projectPoint(frames.back(), 971, 713, 2762.5);
  std::cout << "point:" << point << std::endl;

  std::ifstream map_data_ifs(input + "/map.json");
  nlohmann::json map_data = nlohmann::json::parse(map_data_ifs);
  lar::Map map = map_data;


  frames = nlohmann::json::parse(std::ifstream(input2 + "/frames.json"));

  for (auto frame : frames) {
    char formatted_id[9];
    sprintf(formatted_id, "%08zu", frame.id);
    std::string image_path = input2 + "/" + formatted_id + "_image.jpeg";
    std::string result_path = output + "/" + formatted_id + "_result.jpeg";
    std::string matches_path = output + "/" + formatted_id + "_matches.jpeg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat intrinsics(3, 3, CV_32FC1);
    intrinsics.at<float>(0,0) = frame.intrinsics(0,0);
    intrinsics.at<float>(1,1) = frame.intrinsics(1,1);
    intrinsics.at<float>(0,2) = frame.intrinsics(0,2);
    intrinsics.at<float>(1,2) = frame.intrinsics(1,2);
    intrinsics.at<float>(2,2) = 1.;
    cv::Mat transform;

    lar::Tracker tracker(map);
    std::cout << "parse map" << std::endl;
    tracker.localize(image, intrinsics, transform);
    std::cout << "transform:" << transform << std::endl;



    Eigen::Matrix4d eigen_transform;
    eigen_transform << transform.at<double>(0,0), transform.at<double>(0,1), transform.at<double>(0,2), transform.at<double>(0,3),
                      transform.at<double>(1,0), transform.at<double>(1,1), transform.at<double>(1,2), transform.at<double>(1,3),
                      transform.at<double>(2,0), transform.at<double>(2,1), transform.at<double>(2,2), transform.at<double>(2,3),
                      transform.at<double>(3,0), transform.at<double>(3,1), transform.at<double>(3,2), transform.at<double>(3,3);

    Eigen::Matrix3d eigen_intrinsics;
    eigen_intrinsics << intrinsics.at<float>(0,0), intrinsics.at<float>(0,1), intrinsics.at<float>(0,2),
                        intrinsics.at<float>(1,0), intrinsics.at<float>(1,1), intrinsics.at<float>(1,2),
                        intrinsics.at<float>(2,0), intrinsics.at<float>(2,1), intrinsics.at<float>(2,2);


    cv::Mat result;
    cv::Mat matches;
    std::vector<cv::KeyPoint> kpts = getKeyPoints(eigen_intrinsics, eigen_transform, tracker.local_landmarks);

    cv::drawKeypoints(image, kpts, result);

    Eigen::Vector2d coords = imageCoordinatesFrom(eigen_intrinsics, eigen_transform, point);
    std::cout << "coords:" << coords << std::endl;
    cv::circle(result, cv::Point(coords(0), coords(1)), 50, cv::Scalar(0, 0, 255), 4);

    cv::imwrite(result_path, result);

    cv::drawMatches(image, tracker.kpts, image, kpts, tracker.matches, matches);
    cv::imwrite(matches_path, matches);

  }

  return 0;
}
