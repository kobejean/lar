#ifndef LAR_TRACKING_DETECTORS_GAUSSIAN_H
#define LAR_TRACKING_DETECTORS_GAUSSIAN_H

#include <array>
#include "lar/core/math/math.h"

namespace lar {
  constexpr double SQRT_2_PI = 2.5066282746;
  constexpr double gaussian(int x, double sigma) {
    return (1.0 / (sigma * SQRT_2_PI)) * exp(-x * x / (2.0 * sigma * sigma));
  }

  template <size_t Size>
  constexpr std::array<float, Size> computeGaussianKernel(double sigma) {
    std::array<double, Size> kernel{};
    std::array<float, Size> normalized{};
    int half_size = Size / 2;
    double sum = 0.0;
    for (size_t i = 0; i < Size; ++i) {
      kernel[i] = gaussian(i - half_size, sigma);
      sum += kernel[i];
    }

    for (size_t i = 0; i < Size; ++i) {
      normalized[i] = static_cast<float>(kernel[i] / sum);
    }
    return normalized;
  }

  // template <size_t Size>
  constexpr std::array<std::array<float, 9>, 6> computegaussian_kernels() {
    return std::array<std::array<float, 9>, 6>{
      computeGaussianKernel<9>(1.2699208416), // 1.6 * 2^(-1/3)
      computeGaussianKernel<9>(0.9732939207), // sqrt(1.6^2-1.2699208416^2) = 0.9732939207
      computeGaussianKernel<9>(1.2262734985), // sqrt(2.01587367983^2-1.6^2) = 1.2262734985
      computeGaussianKernel<9>(1.5450077936), // sqrt(2.5398416831^2-2.01587367983^2) = 1.5450077936
      computeGaussianKernel<9>(1.9465878415), // sqrt(3.2^2-2.5398416831^2) = 1.9465878415
      computeGaussianKernel<9>(2.4525469969)  // sqrt(4.03174735966^2-3.2^2) = 2.4525469969
    };
  }

  // ⌈InvNorm(1-0.04)*σ_2⌉ is an upperbound solution to (solving for x):
  // t = 0.04 > Φ(x/σ_1) - Φ(x/σ_2), where x > 0
  constexpr int computeInvalidationSize(double sigma) {
    const double invNorm = 1.750686; // InvNorm(1-0.04)
    return ceil(invNorm * sigma);
  }

  template <size_t num_scales>
  constexpr std::array<int, num_scales> computeInvalidationSizes() {
    std::array<int, num_scales> result; // Create an array instance named result
    for (size_t i = 1; i <= num_scales; i++) {
      double exponent = static_cast<double>(i) / static_cast<double>(num_scales);
      result[i-1] = computeInvalidationSize(1.6 * pow(2, exponent));
    }
    return result;
  }
}

#endif /* LAR_TRACKING_DETECTORS_GAUSSIAN_H */