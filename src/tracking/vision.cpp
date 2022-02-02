
#include "lar/tracking/vision.h"

namespace lar {

  const float RATIO_TEST_THRESHOLD = 0.6f;
  const float MARGIN_TEST_DISTANCE = 25.f; // TODO: Think of clearer name

  Vision::Vision() {
    detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 4, 4, cv::KAZE::DIFF_PM_G2);
    matcher = cv::BFMatcher(cv::NORM_HAMMING);
  }

  void Vision::extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    std::vector<cv::KeyPoint> new_kpts;
    cv::Mat new_desc;

    detector->detectAndCompute(image, mask, new_kpts, new_desc);

    // Match function uses ratio test and margin test. We can reuse it to filter features 
    // that are not distinct enough by matching the set of descriptions with each other.
    std::vector<cv::DMatch> matches = match(new_desc, new_desc);

    // Populate `desc` and `kpts` with matched key points and descriptions
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      // Matches should have sucessfully matched with itself, but it is redundant to check if it did
      assert(idx == matches[i].trainIdx);
      kpts.push_back(new_kpts[idx]);
      desc.push_back(new_desc.row(idx));
    }
  }

  std::vector<cv::DMatch> Vision::match(const cv::Mat& desc1, const cv::Mat& desc2) {
    std::vector<cv::DMatch> filtered_matches;
    // We need at least 2 rows to perform ratio test
    if (desc1.rows <= 2 || desc2.rows <= 2) return filtered_matches;
    std::vector< std::vector<cv::DMatch> > nn_matches;

    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    // Filter matches
    for (size_t i = 0; i < nn_matches.size(); i++) {
      cv::DMatch first = nn_matches[i][0];
      float dist1 = nn_matches[i][0].distance;
      float dist2 = nn_matches[i][1].distance;

      bool ratio_test = dist1 < RATIO_TEST_THRESHOLD * dist2;
      bool margin_test = dist2 >= MARGIN_TEST_DISTANCE;
      if (ratio_test && margin_test) {
        filtered_matches.push_back(first);
      }
    }

    return filtered_matches;
  }

}