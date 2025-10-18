#ifndef LAR_TRACKING_MATCHING_BF_MATCHER_METAL_H
#define LAR_TRACKING_MATCHING_BF_MATCHER_METAL_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace lar {

/**
 * @brief Metal-accelerated brute-force descriptor matcher optimized for LAR's feature matching.
 *
 * This matcher uses Metal compute shaders to perform parallel brute-force matching on the GPU.
 * It's optimized for:
 * - SIFT uint8 descriptors (128-dimensional)
 * - Fixed k=8 nearest neighbor search
 * - L2 distance metric
 * - Real-time AR localization performance
 *
 * Unlike FLANN's approximate matching, this provides exact matches using exhaustive search,
 * but accelerated via GPU parallelism.
 */
class BFMatcherMetal {
public:
    /**
     * @brief Constructor that initializes Metal resources.
     * @throws std::runtime_error if Metal device or shader compilation fails
     */
    BFMatcherMetal();

    /**
     * @brief Destructor that releases Metal resources.
     */
    ~BFMatcherMetal();

    // Disable copy (Metal resources are not copyable)
    BFMatcherMetal(const BFMatcherMetal&) = delete;
    BFMatcherMetal& operator=(const BFMatcherMetal&) = delete;

    // Allow move operations
    BFMatcherMetal(BFMatcherMetal&& other) noexcept;
    BFMatcherMetal& operator=(BFMatcherMetal&& other) noexcept;

    /**
     * @brief Find k=8 nearest matches for each query descriptor using Metal compute.
     *
     * This method performs brute-force matching on the GPU:
     * 1. Uploads query and train descriptors to Metal buffers (if needed)
     * 2. Dispatches compute shader to calculate all pairwise distances
     * 3. Finds top-8 closest matches per query using partial sort
     * 4. Downloads results back to CPU
     *
     * @param queryDescriptors Query descriptor set (rows = descriptors, must be CV_8U or CV_32F)
     * @param trainDescriptors Train descriptor set (rows = descriptors, must match query type)
     * @param matches Output matches (matches[i] = 8 best matches for query i)
     * @param k Number of best matches (fixed to 8 for optimization, parameter kept for API compatibility)
     *
     * @note Descriptors must be contiguous (isContinuous() == true)
     * @note Currently supports L2 distance only
     * @note Both descriptor sets must have the same number of columns
     */
    void knnMatch(
        const cv::Mat& queryDescriptors,
        const cv::Mat& trainDescriptors,
        std::vector<std::vector<cv::DMatch>>& matches,
        int k = 8
    );

    /**
     * @brief Check if Metal is available and matcher is ready.
     * @return true if Metal resources initialized successfully
     */
    bool isReady() const;

    /**
     * @brief Get performance statistics (GPU time, transfer time, etc.)
     * @return String with timing breakdown
     */
    std::string getPerformanceStats() const;

private:
    class Impl;  // Forward declaration for pImpl idiom (hides Metal types from header)
    Impl* impl_;
};

} // namespace lar

#endif // LAR_TRACKING_MATCHING_BF_MATCHER_METAL_H