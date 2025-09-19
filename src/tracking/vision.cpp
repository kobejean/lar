
#include "lar/tracking/vision.h"

#include <iostream>
namespace lar {

  const float RATIO_TEST_THRESHOLD = 0.99f;
  const float MAX_DISTANCE_THRESHOLD = 210.0f;
  const bool ENABLE_CROSS_CHECK = false; // seems best to use only for image to image matching

  Vision::Vision() {
    detector = cv::SIFT::create(0,3,0.02,10,1.6,CV_8U);

    // Use FLANN matcher for better performance with large datasets
    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
    flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);

    // Keep BFMatcher as fallback
    bf_matcher = cv::BFMatcher(cv::NORM_L2, false);
  }

  void Vision::extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    size_t max_features = 8192*2;
    float min_scale = 2.9;
    detector->detectAndCompute(image, mask, kpts, desc);

    // Filter out feature points that are too small 
    std::vector<std::pair<float, int>> scale_indices;
    for (int i = 0; i < kpts.size(); ++i) {
      if (kpts[i].size >= min_scale) {
        scale_indices.emplace_back(kpts[i].size, i);
      }
    }
    
    // Sort by scale in descending order (largest first)
    std::sort(scale_indices.begin(), scale_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { 
                return a.first > b.first; 
              });

    // Keep only top max_features from the scale-filtered results
    std::vector<cv::KeyPoint> filtered_kpts;
    cv::Mat filtered_desc;
    size_t num_features = std::min(scale_indices.size(), max_features);
    filtered_kpts.reserve(num_features);
    
    if (num_features > 0 && desc.rows > 0) {
      filtered_desc.create(num_features, desc.cols, desc.type());
      
      for (size_t i = 0; i < num_features; ++i) {
        int original_idx = scale_indices[i].second;
        filtered_kpts.push_back(kpts[original_idx]);
        desc.row(original_idx).copyTo(filtered_desc.row(i));
      }
    }

    // Modify the reference parameters
    kpts = std::move(filtered_kpts);
    desc = filtered_desc.clone(); // Use clone() to ensure proper matrix copy

    std::cout << "Filtered to " << kpts.size() << " largest-scale features (scale >= " << min_scale << ") "
              << "(scale range: " << (scale_indices.empty() ? 0.0f : scale_indices[0].first) << " - " 
              << (num_features > 0 ? scale_indices[num_features-1].first : 0.0f) << ")" << std::endl;
  }

  std::vector<cv::DMatch> Vision::matchOneWay(const cv::Mat& desc1, const cv::Mat& desc2) const {
    std::vector<cv::DMatch> filtered_matches;
    if (desc1.rows <= 2 || desc2.rows <= 2) return filtered_matches;

    std::vector<std::vector<cv::DMatch>> nn_matches;

    // Convert uint8 descriptors to float32 for FLANN
    cv::Mat desc1_float, desc2_float;
    desc1.convertTo(desc1_float, CV_32F);
    desc2.convertTo(desc2_float, CV_32F);

    flann_matcher.knnMatch(desc1_float, desc2_float, nn_matches, 5);
    std::sort(nn_matches.begin(), nn_matches.end(), [](const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b) {
      if (a.size() < 2 || b.size() < 2) return a.size() > b.size();
      return a[0].distance * b[1].distance < a[1].distance * b[0].distance;
    });
    std::vector<cv::DMatch> distance_matches;
    for (const auto& nn_match : nn_matches) {
      float dist1 = nn_match[0].distance;
      if (dist1 < MAX_DISTANCE_THRESHOLD && nn_match.size() >= 2 && dist1 < RATIO_TEST_THRESHOLD * nn_match[1].distance) {
        // Passed distance and ratio test
        filtered_matches.push_back(nn_match[0]);
      } else {
        for (auto& match : nn_match) {
          if (match.distance > MAX_DISTANCE_THRESHOLD || dist1 < RATIO_TEST_THRESHOLD * match.distance) break;
          // these fail ratio test but might still help pnp
          distance_matches.push_back(match);
        }
      }
    }
    std::cout << "ratio passed: " << filtered_matches.size() << std::endl;
    
    std::sort(distance_matches.begin(), distance_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
      return a.distance < b.distance;
    });

    filtered_matches.insert(filtered_matches.end(), 
                       std::make_move_iterator(distance_matches.begin()), 
                       std::make_move_iterator(distance_matches.end()));
    // if (filtered_matches.size() > 3000) filtered_matches.resize(3000);
    return filtered_matches;
  }

  std::vector<cv::DMatch> Vision::match(const cv::Mat& desc1, const cv::Mat& desc2) {
    std::vector<cv::DMatch> final_matches;
    if (desc1.rows <= 2 || desc2.rows <= 2) return final_matches;

    std::vector<cv::DMatch> matches_12 = matchOneWay(desc1, desc2);

    if (!ENABLE_CROSS_CHECK) return matches_12;
    
    std::vector<cv::DMatch> matches_21 = matchOneWay(desc2, desc1);
    
    // Cross-check: keep only bidirectional matches
    for (const auto& match_12 : matches_12) {
      for (const auto& match_21 : matches_21) {
        if (match_12.queryIdx == match_21.trainIdx && 
            match_12.trainIdx == match_21.queryIdx) {
          final_matches.push_back(match_12);
          break;
        }
      }
    }
    
    return final_matches;
  }
}
