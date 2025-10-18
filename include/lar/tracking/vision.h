#ifndef LAR_TRACKING_VISION_H
#define LAR_TRACKING_VISION_H

#include "sift/sift.h"
#include "matching/flann_matcher.h"
#include "matching/bf_matcher_metal.h"
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/flann.hpp>

namespace lar {

  class Vision {
    public:
      cv::Ptr<SIFT> detector;
      FlannMatcher flann_matcher;

      /// Constructor with optional image dimensions for Metal SIFT optimization
      /// @param imageSize Expected input image dimensions (default: 1920x1440, typical ARKit resolution)
      Vision(cv::Size imageSize = cv::Size(1920, 1440));

      // Delete copy operations (SIFT is not copyable)
      Vision(const Vision&) = delete;
      Vision& operator=(const Vision&) = delete;

      // Allow move operations
      Vision(Vision&&) = default;
      Vision& operator=(Vision&&) = default;

      /// Reconfigure detector with new image dimensions
      /// @param imageSize New expected input image dimensions (width, height)
      void configureImageSize(cv::Size imageSize);

      void extractFeatures(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      std::vector<cv::DMatch> match(const cv::Mat &desc1, const cv::Mat &desc2,
                                     const std::vector<cv::KeyPoint>& kpts);

    private:
      std::vector<cv::DMatch> matchOneWay(const cv::Mat& desc1, const cv::Mat& desc2,
                                           const std::vector<cv::KeyPoint>& kpts);
  };

}

#endif /* LAR_TRACKING_VISION_H */