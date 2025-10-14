// Metal implementation of SIFT::Impl (Pimpl idiom)
// This file should only be included by sift.cpp when LAR_USE_METAL_SIFT is defined
// Note: This header is designed to be compiled as Objective-C++ (.mm file only)
#ifndef LAR_TRACKING_SIFT_IMPL_METAL_H
#define LAR_TRACKING_SIFT_IMPL_METAL_H

#include "lar/tracking/sift/sift.h"

// This header declares SIFT::Impl for Metal implementation
// The actual struct definition with Metal types is in the .mm file
// This allows sift.cpp to forward-declare the implementation without knowing Metal types

namespace lar {

// Forward declaration - actual definition is in sift_impl_metal.mm
// We can't define the struct here because it contains Objective-C types

} // namespace lar

#endif // LAR_TRACKING_SIFT_IMPL_METAL_H