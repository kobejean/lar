#ifndef LAR_TRACKING_DETECTORS_SIFT_H
#define LAR_TRACKING_DETECTORS_SIFT_H

#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include "lar/tracking/detectors/gaussian.h"

namespace lar {

  // template <int num_scales>
  class SIFT {
    public:
      SIFT();
      void detect(cv::InputArray image, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

      template <size_t kernel_size>
      void processOctave(const cv::Mat& img, int octave);
      void computeDoG(const cv::Mat& img);
      void extractDescriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, int start_idx);
      void extractFeatures(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);
    
      std::array<std::array<cv::Mat, 6>, 5> gaussians;
      std::array<std::array<cv::Mat, 5>, 5> DoG;
      static constexpr int num_scales = 3;
      static constexpr int num_octaves = 5;
      static constexpr uchar contrast_threshold = static_cast<uchar>(0.04 * 256);
  };

}

#endif /* LAR_TRACKING_DETECTORS_SIFT_H */