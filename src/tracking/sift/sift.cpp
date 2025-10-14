// SIFT implementation using Pimpl idiom
// This file delegates to either CPU or Metal implementation based on compile-time flags

#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_common.h"

// Conditionally include the appropriate implementation
// Important: Must include the full implementation, not just forward declarations
#ifdef LAR_USE_METAL_SIFT
    // For Metal, the implementation is in the .mm file
    // We only need the forward declaration here
#else
    #include "sift_impl_cpu.h"
#endif

namespace lar {

// SIFT public API implementation - delegates to Pimpl

SIFT::SIFT(const SIFTConfig& config)
    : impl_(new Impl(config))
{
}

SIFT::~SIFT() {
    delete impl_;
}

SIFT::SIFT(SIFT&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

SIFT& SIFT::operator=(SIFT&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

int SIFT::descriptorSize() const {
    return impl_->descriptorSize();
}

int SIFT::descriptorType() const {
    return impl_->descriptorType();
}

int SIFT::defaultNorm() const {
    return impl_->defaultNorm();
}

void SIFT::detectAndCompute(cv::InputArray image, cv::InputArray mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray descriptors,
                            bool useProvidedKeypoints) {
    impl_->detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints);
}

} // namespace lar