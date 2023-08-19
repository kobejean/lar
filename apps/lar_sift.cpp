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

void visualizeKeyPoints(std::string filepath, const cv::Mat& img, std::vector<cv::KeyPoint>& kpts) {
  cv::Mat output;
  cv::drawKeypoints(img, kpts, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imwrite(filepath, output);
}

void visualizeMatches(std::string filepath, const cv::Mat& img1, std::vector<cv::KeyPoint>& kpts1,
                                            const cv::Mat& img2, std::vector<cv::KeyPoint>& kpts2,
                                            const std::vector<cv::DMatch>& matches) {
  // Draw and display the matches
  cv::Mat output;
  cv::drawMatches(img1, kpts1, img2, kpts2, matches, output);
  cv::imwrite(filepath, output);
}

std::vector<cv::DMatch> match(cv::Mat& desc1, cv::Mat& desc2) {
  // Use the BFMatcher to find the two best matches for each descriptor
  cv::BFMatcher matcher(cv::NORM_L2); // L2 norm for SIFT
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher.knnMatch(desc1, desc2, knnMatches, 2); // k = 2

  // Apply the ratio test
  float ratio_thresh = 0.75f; // Adjust as needed
  std::vector<cv::DMatch> goodMatches;
  for (const auto& knnMatch : knnMatches) {
    if (knnMatch[0].distance < ratio_thresh * knnMatch[1].distance) {
      goodMatches.push_back(knnMatch[0]);
    }
  }
  return goodMatches;
}

int main(int argc, const char* argv[]){
  string input = "./input/snapshot";

  struct stat st;
  int status = stat(input.c_str(), &st);

  if (status != 0) {
    cout << "Could not read directory at '" << input << endl;
    return 1;
  }

  cv::Mat img1 = cv::imread(input + "/00000004_image.jpeg", cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(input + "/00000003_image.jpeg", cv::IMREAD_GRAYSCALE);

  lar::SIFT sift;
  std::vector<cv::KeyPoint> kpts1, kpts2;
  cv::Mat desc1, desc2;

  sift.detect(img1, kpts1, desc1);
  visualizeKeyPoints("./output/sift/kpts1.jpeg", img1, kpts1);

  sift.detect(img2, kpts2, desc2);
  visualizeKeyPoints("./output/sift/kpts2.jpeg", img2, kpts2);

  auto matches = match(desc1, desc2);
  visualizeMatches("./output/sift/matches.jpeg", img1, kpts1, img2, kpts2, matches);

  std::cout << "kpts1: " << kpts1.size() << std::endl;
  std::cout << "kpts2: " << kpts2.size() << std::endl;
  std::cout << "matches: " << matches.size() << std::endl;
  return 0;
}
