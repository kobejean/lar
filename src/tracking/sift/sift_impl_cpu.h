// CPU implementation of SIFT::Impl (Pimpl idiom)
// This file should only be included by sift.cpp
#ifndef LAR_TRACKING_SIFT_IMPL_CPU_H
#define LAR_TRACKING_SIFT_IMPL_CPU_H

#include <opencv2/core.hpp>
#include <vector>
#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_common.h"

namespace lar {

// CPU-only implementation of SIFT using OpenCV and optional SIMD
struct SIFT::Impl {
    SIFTConfig config;

    explicit Impl(const SIFTConfig& cfg);
    ~Impl() = default;

    // Delete copy operations
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Allow move operations
    Impl(Impl&& other) noexcept = default;
    Impl& operator=(Impl&& other) noexcept = default;

    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;

private:
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
    void buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const;
    void findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                              const std::vector<cv::Mat>& dog_pyr,
                              std::vector<cv::KeyPoint>& keypoints) const;
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_IMPL_CPU_H