#include "lar/tracking/matching/flann_matcher.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace lar {

// ============================================================================
// DescriptorCollection Implementation
// ============================================================================

FlannMatcher::DescriptorCollection::DescriptorCollection() {}

FlannMatcher::DescriptorCollection::DescriptorCollection(const DescriptorCollection& collection) {
    mergedDescriptors = collection.mergedDescriptors.clone();
    startIdxs = collection.startIdxs;
}

FlannMatcher::DescriptorCollection::~DescriptorCollection() {}

void FlannMatcher::DescriptorCollection::set(const std::vector<cv::Mat>& descriptors) {
    clear();

    size_t imageCount = descriptors.size();
    if (imageCount == 0) return;

    startIdxs.resize(imageCount);

    int dim = -1;
    int type = -1;
    startIdxs[0] = 0;

    for (size_t i = 1; i < imageCount; i++) {
        int s = 0;
        if (!descriptors[i-1].empty()) {
            dim = descriptors[i-1].cols;
            type = descriptors[i-1].type();
            s = descriptors[i-1].rows;
        }
        startIdxs[i] = startIdxs[i-1] + s;
    }

    if (imageCount == 1) {
        if (descriptors[0].empty()) return;
        dim = descriptors[0].cols;
        type = descriptors[0].type();
    }

    if (dim <= 0) return;

    int count = startIdxs[imageCount-1] + descriptors[imageCount-1].rows;

    if (count > 0) {
        mergedDescriptors.create(count, dim, type);
        for (size_t i = 0; i < imageCount; i++) {
            if (!descriptors[i].empty()) {
                cv::Mat m = mergedDescriptors.rowRange(startIdxs[i], startIdxs[i] + descriptors[i].rows);
                descriptors[i].copyTo(m);
            }
        }
    }
}

void FlannMatcher::DescriptorCollection::clear() {
    startIdxs.clear();
    mergedDescriptors.release();
}

cv::Mat FlannMatcher::DescriptorCollection::getDescriptor(int imgIdx, int localDescIdx) const {
    if (imgIdx >= (int)startIdxs.size()) {
        throw std::out_of_range("Image index out of range");
    }
    int globalIdx = startIdxs[imgIdx] + localDescIdx;
    if (globalIdx >= size()) {
        throw std::out_of_range("Global descriptor index out of range");
    }
    return getDescriptor(globalIdx);
}

cv::Mat FlannMatcher::DescriptorCollection::getDescriptor(int globalDescIdx) const {
    if (globalDescIdx < 0 || globalDescIdx >= size()) {
        throw std::out_of_range("Descriptor index out of range");
    }
    return mergedDescriptors.row(globalDescIdx);
}

void FlannMatcher::DescriptorCollection::getLocalIdx(int globalDescIdx, int& imgIdx, int& localDescIdx) const {
    if (globalDescIdx < 0 || globalDescIdx >= size()) {
        throw std::out_of_range("Global descriptor index out of range");
    }

    std::vector<int>::const_iterator img_it = std::upper_bound(startIdxs.begin(), startIdxs.end(), globalDescIdx);
    --img_it;
    imgIdx = (int)(img_it - startIdxs.begin());
    localDescIdx = globalDescIdx - (*img_it);
}

// ============================================================================
// FlannMatcher Implementation
// ============================================================================

FlannMatcher::FlannMatcher(
    const cv::Ptr<cv::flann::IndexParams>& indexParams_,
    const cv::Ptr<cv::flann::SearchParams>& searchParams_
) : indexParams(indexParams_), searchParams(searchParams_), addedDescCount(0) {
    if (!indexParams) {
        throw std::invalid_argument("indexParams cannot be null");
    }
    if (!searchParams) {
        throw std::invalid_argument("searchParams cannot be null");
    }
}

FlannMatcher::~FlannMatcher() {}

void FlannMatcher::add(const cv::Mat& descriptors) {
    if (!descriptors.empty()) {
        trainDescCollection.push_back(descriptors);
        addedDescCount += descriptors.rows;
    }
}

void FlannMatcher::add(const std::vector<cv::Mat>& descriptors) {
    for (const auto& desc : descriptors) {
        add(desc);
    }
}

void FlannMatcher::clear() {
    trainDescCollection.clear();
    mergedDescriptors.clear();
    flannIndex.release();
    addedDescCount = 0;
}

bool FlannMatcher::empty() const {
    return trainDescCollection.empty();
}

void FlannMatcher::train() {
    if (!flannIndex || mergedDescriptors.size() < addedDescCount) {
        mergedDescriptors.set(trainDescCollection);
        if (mergedDescriptors.size() > 0) {
            flannIndex = cv::makePtr<cv::flann::Index>(
                mergedDescriptors.getDescriptors(),
                *indexParams
            );
        }
    }
}

void FlannMatcher::knnMatch(
    const cv::Mat& queryDescriptors,
    std::vector<std::vector<cv::DMatch>>& matches,
    int k
) {
    if (empty() || queryDescriptors.empty()) {
        matches.clear();
        return;
    }

    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    // Ensure the index is trained
    train();

    if (!flannIndex) {
        matches.clear();
        return;
    }

    // Prepare output matrices
    cv::Mat indices(queryDescriptors.rows, k, CV_32SC1);
    cv::Mat dists(queryDescriptors.rows, k, CV_32FC1);

    // Perform FLANN search
    flannIndex->knnSearch(queryDescriptors, indices, dists, k, *searchParams);

    // Convert to DMatch format
    convertToDMatches(mergedDescriptors, indices, dists, matches);
}

void FlannMatcher::convertToDMatches(
    const DescriptorCollection& collection,
    const cv::Mat& indices,
    const cv::Mat& dists,
    std::vector<std::vector<cv::DMatch>>& matches
) {
    matches.resize(indices.rows);

    for (int i = 0; i < indices.rows; i++) {
        matches[i].clear();

        for (int j = 0; j < indices.cols; j++) {
            int idx = indices.at<int>(i, j);

            if (idx >= 0) {
                int imgIdx, trainIdx;
                collection.getLocalIdx(idx, imgIdx, trainIdx);

                float dist = 0.0f;
                if (dists.type() == CV_32S) {
                    dist = static_cast<float>(dists.at<int>(i, j));
                } else {
                    // FLANN returns squared distances, so take sqrt
                    dist = std::sqrt(dists.at<float>(i, j));
                }

                matches[i].push_back(cv::DMatch(i, trainIdx, imgIdx, dist));
            }
        }
    }
}

} // namespace lar