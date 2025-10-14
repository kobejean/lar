#ifndef LAR_TRACKING_SIFT_H
#define LAR_TRACKING_SIFT_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>
#include "lar/tracking/sift/sift_common.h"

// Forward declaration for SIFTMetal (only when Metal is enabled)
#define LAR_USE_METAL_SIFTO 1
#ifdef LAR_USE_METAL_SIFTO
namespace lar {
    class SIFTMetal;
}
#endif

namespace lar {

class SIFT {
public:
    SIFT(const SIFTConfig& config);
    ~SIFT();

    // Delete copy operations (SIFTMetal is not copyable)
    SIFT(const SIFT&) = delete;
    SIFT& operator=(const SIFT&) = delete;

    // Allow move operations
    SIFT(SIFT&& other) noexcept;
    SIFT& operator=(SIFT&& other) noexcept;

    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints = false);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;

private:
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
    void buildDoGPyramid(const std::vector<cv::Mat>& pyr, std::vector<cv::Mat>& dogpyr) const;
    void findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                               const std::vector<cv::Mat>& dog_pyr,
                               std::vector<cv::KeyPoint>& keypoints) const;

    SIFTConfig config_;

#ifdef LAR_USE_METAL_SIFTO
    std::unique_ptr<SIFTMetal> metalSift_;
#endif
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_H