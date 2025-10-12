#include "lar/tracking/vision.h"

#include <iostream>
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/imgproc.hpp>

namespace lar {

  const float RATIO_TEST_THRESHOLD = 0.99f;
  const float MAX_DISTANCE_THRESHOLD = 210.0f;

  Vision::Vision() {
    detector = SIFT::create(0, 3, 0.02, 10, 1.6, CV_8U);
    // detector = cv::SIFT::create(0, 3, 0.02, 10, 1.6, CV_8U);

    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(3);
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(32);
    flann_matcher = cv::FlannBasedMatcher(indexParams, searchParams);
    bf_matcher = cv::BFMatcher(cv::NORM_L2, false);
  }

  void Vision::extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    auto start_extract = std::chrono::high_resolution_clock::now();

    size_t max_features = 8192*2;

    auto start_sift = std::chrono::high_resolution_clock::now();
    detector->detectAndCompute(image, mask, kpts, desc);
    auto end_sift = std::chrono::high_resolution_clock::now();

    // Filter out feature points that are too small
    std::vector<std::pair<float, int>> scale_indices;
    for (size_t i = 0; i < kpts.size(); ++i) {
      scale_indices.emplace_back(kpts[i].size, i);
    }

    // Sort by scale in descending order (largest first)
    std::sort(scale_indices.begin(), scale_indices.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

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

    auto end_extract = std::chrono::high_resolution_clock::now();

    // Log timing information
    auto sift_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_sift - start_sift).count();
    auto total_extract_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_extract - start_extract).count();

    std::cout << "Feature Extraction - SIFT: " << sift_time << "ms, Total: " << total_extract_time << std::endl;
    std::cout << "Filtered to " << kpts.size() << " largest-scale features "
              << "(scale range: " << (scale_indices.empty() ? 0.0f : scale_indices[0].first) << " - "
              << (num_features > 0 ? scale_indices[num_features-1].first : 0.0f) << ")" << std::endl;
  }

  std::vector<cv::DMatch> Vision::matchOneWay(const cv::Mat& desc1, const cv::Mat& desc2,
                                               const std::vector<cv::KeyPoint>& kpts) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<cv::DMatch> filtered_matches;
    if (desc1.rows <= 2 || desc2.rows <= 2) return filtered_matches;

    // Pre-allocate result vectors to reduce memory allocation overhead
    std::vector<std::vector<cv::DMatch>> nn_matches;
    nn_matches.reserve(desc1.rows);
    filtered_matches.reserve(desc1.rows / 4); // Conservative estimate

    // Convert uint8 descriptors to float32 for FLANN (SIFT descriptors work best with KD-tree)
    auto start_convert = std::chrono::high_resolution_clock::now();
    cv::Mat desc1_float, desc2_float;
    desc1.convertTo(desc1_float, CV_32F);
    desc2.convertTo(desc2_float, CV_32F);
    auto end_convert = std::chrono::high_resolution_clock::now();

    // Use optimized KD-tree parameters - faster than LSH for SIFT descriptors
    auto start_match = std::chrono::high_resolution_clock::now();

    flann_matcher.knnMatch(desc1_float, desc2_float, nn_matches, 8);
    auto end_match = std::chrono::high_resolution_clock::now();
    std::sort(nn_matches.begin(), nn_matches.end(), [](const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b) {
      if (a.size() < 2 || b.size() < 2) return a.size() > b.size();
      return a[0].distance * b[1].distance < a[1].distance * b[0].distance;
    });
    std::vector<cv::DMatch> distance_matches;
    distance_matches.reserve(desc1.rows / 8); // Conservative estimate for distance matches
    for (const auto& nn_match : nn_matches) {
      if (nn_match.empty()) continue;

      float dist1 = nn_match[0].distance;
      if (dist1 < MAX_DISTANCE_THRESHOLD && nn_match.size() >= 2 && dist1 < RATIO_TEST_THRESHOLD * nn_match[1].distance) {
        // Passed distance and ratio test
        filtered_matches.push_back(nn_match[0]);
      } else {
        for (const auto& match : nn_match) {
          if (match.distance > MAX_DISTANCE_THRESHOLD || dist1 < RATIO_TEST_THRESHOLD * match.distance) break;
          // these fail ratio test but might still help pnp
          distance_matches.push_back(match);
        }
      }
    }
    std::cout << "ratio passed: " << filtered_matches.size() << std::endl;

    const float SIZE_THRESHOLD = 3.5f;

    std::sort(distance_matches.begin(), distance_matches.end(),
      [&kpts, SIZE_THRESHOLD](const cv::DMatch& a, const cv::DMatch& b) {
        // return a.distance < b.distance;
        float size_a = kpts[a.queryIdx].size;
        float size_b = kpts[b.queryIdx].size;

        // If both are large features, sort by distance
        if (size_a > SIZE_THRESHOLD && size_b > SIZE_THRESHOLD) {
          return a.distance < b.distance;
        }

        // Otherwise, sort by size (larger first)
        return size_a > size_b;
      });

    filtered_matches.insert(filtered_matches.end(),
                       std::make_move_iterator(distance_matches.begin()),
                       std::make_move_iterator(distance_matches.end()));

    auto end_total = std::chrono::high_resolution_clock::now();

    // Log timing information
    auto convert_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - start_convert).count();
    auto match_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_match - start_match).count();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

    std::cout << "Feature Matching - Convert: " << convert_time << "ms, FLANN: " << match_time << "ms, Total: " << total_time << "ms" << std::endl;

    return filtered_matches;
  }

  std::vector<cv::DMatch> Vision::match(const cv::Mat& desc1, const cv::Mat& desc2,
                                         const std::vector<cv::KeyPoint>& kpts) {
    if (desc1.rows <= 2 || desc2.rows <= 2) return {};

    // Use simple one-way matching
    return matchOneWay(desc1, desc2, kpts);
  }
}