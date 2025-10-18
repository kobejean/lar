#ifndef LAR_TRACKING_MATCHING_FLANN_MATCHER_H
#define LAR_TRACKING_MATCHING_FLANN_MATCHER_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <vector>

namespace lar {

/**
 * @brief Custom FLANN-based descriptor matcher optimized for LAR's feature matching.
 *
 * This matcher is based on OpenCV's FlannBasedMatcher but customized for:
 * - SIFT uint8 descriptors (128-dimensional)
 * - Performance optimization for AR localization
 * - Custom filtering and result processing
 *
 * Like the original, this matcher trains cv::flann::Index on train descriptors
 * and uses its nearest search methods. It does not support masking permissible
 * matches because flann::Index does not support this.
 */
class FlannMatcher {
public:
    /**
     * @brief Constructor with index and search parameters.
     * @param indexParams FLANN index parameters (default: KDTree with 4 trees)
     * @param searchParams FLANN search parameters (default: 32 max checks)
     */
    FlannMatcher(
        const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(4),
        const cv::Ptr<cv::flann::SearchParams>& searchParams = cv::makePtr<cv::flann::SearchParams>(32)
    );

    virtual ~FlannMatcher();

    /**
     * @brief Add descriptors to the training set.
     * @param descriptors Descriptors to add (each row is one descriptor)
     */
    void add(const cv::Mat& descriptors);

    /**
     * @brief Add multiple descriptor sets to the training set.
     * @param descriptors Vector of descriptor matrices
     */
    void add(const std::vector<cv::Mat>& descriptors);

    /**
     * @brief Clear all training descriptors.
     */
    void clear();

    /**
     * @brief Check if the matcher has no training descriptors.
     * @return true if empty
     */
    bool empty() const;

    /**
     * @brief Train the FLANN index with accumulated descriptors.
     * Called automatically before matching if needed.
     */
    void train();

    /**
     * @brief Find k nearest matches for each query descriptor.
     * @param queryDescriptors Query descriptor set (rows = descriptors)
     * @param matches Output matches (matches[i] = k best matches for query i)
     * @param k Number of best matches to find per query descriptor
     */
    void knnMatch(
        const cv::Mat& queryDescriptors,
        std::vector<std::vector<cv::DMatch>>& matches,
        int k
    );

    /**
     * @brief Find all matches within a given radius.
     * @param queryDescriptors Query descriptor set
     * @param matches Output matches
     * @param maxDistance Maximum distance threshold
     */
    void radiusMatch(
        const cv::Mat& queryDescriptors,
        std::vector<std::vector<cv::DMatch>>& matches,
        float maxDistance
    );

    /**
     * @brief Get the training descriptors.
     * @return Reference to training descriptor vector
     */
    const std::vector<cv::Mat>& getTrainDescriptors() const { return trainDescCollection; }

private:
    /**
     * @brief Helper class to manage merged descriptors from multiple images.
     * Used internally to combine descriptor sets for FLANN indexing.
     */
    class DescriptorCollection {
    public:
        DescriptorCollection();
        DescriptorCollection(const DescriptorCollection& collection);
        virtual ~DescriptorCollection();

        // Merge multiple descriptor matrices into one
        void set(const std::vector<cv::Mat>& descriptors);
        virtual void clear();

        // Accessors
        const cv::Mat& getDescriptors() const { return mergedDescriptors; }
        cv::Mat getDescriptor(int imgIdx, int localDescIdx) const;
        cv::Mat getDescriptor(int globalDescIdx) const;
        void getLocalIdx(int globalDescIdx, int& imgIdx, int& localDescIdx) const;
        int size() const { return mergedDescriptors.rows; }

    private:
        cv::Mat mergedDescriptors;      // All descriptors merged into one matrix
        std::vector<int> startIdxs;     // Start index for each image's descriptors
    };

    /**
     * @brief Convert FLANN search results to DMatch format.
     * @param collection Descriptor collection for index mapping
     * @param indices FLANN indices matrix
     * @param dists FLANN distances matrix
     * @param matches Output matches in DMatch format
     */
    static void convertToDMatches(
        const DescriptorCollection& collection,
        const cv::Mat& indices,
        const cv::Mat& dists,
        std::vector<std::vector<cv::DMatch>>& matches
    );

    // FLANN parameters
    cv::Ptr<cv::flann::IndexParams> indexParams;
    cv::Ptr<cv::flann::SearchParams> searchParams;
    cv::Ptr<cv::flann::Index> flannIndex;

    // Training data
    std::vector<cv::Mat> trainDescCollection;
    DescriptorCollection mergedDescriptors;
    int addedDescCount;
};

} // namespace lar

#endif // LAR_TRACKING_MATCHING_FLANN_MATCHER_H