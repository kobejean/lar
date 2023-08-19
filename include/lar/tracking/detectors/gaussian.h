#ifndef LAR_TRACKING_DETECTORS_GAUSSIAN_H
#define LAR_TRACKING_DETECTORS_GAUSSIAN_H

#include <array>
#include "lar/core/math/math.h"

namespace lar {
  constexpr double SQRT_2_PI = 2.5066282746;
  constexpr double gaussian(double x, double sigma) {
    return (1.0 / (sigma * SQRT_2_PI)) * exp(-x * x / (2.0 * sigma * sigma));
  }

  template <size_t Size>
  constexpr std::array<float, Size> computeGaussianKernel(double sigma, double center) {
    std::array<double, Size> kernel{};
    std::array<float, Size> normalized{};
    double sum = 0.0;
    for (size_t i = 0; i < Size; ++i) {
      double x = static_cast<double>(i) + 0.5 - center;
      kernel[i] = gaussian(x, sigma);
      sum += kernel[i];
    }

    for (size_t i = 0; i < Size; ++i) {
      normalized[i] = static_cast<float>(kernel[i] / sum);
    }
    return normalized;
  }

  template <size_t kernel_size>
  constexpr std::array<float, kernel_size> computeGaussianKernel(double sigma) {
    double center = static_cast<double>(kernel_size) / 2.0;
    return computeGaussianKernel<kernel_size>(sigma, center);
  }

  template <size_t kernel_size>
  constexpr std::array<std::array<float, kernel_size>, 6> computeGaussianKernels() {
    return std::array<std::array<float, kernel_size>, 6>{
      computeGaussianKernel<kernel_size>(1.2699208416), // 1.6 * 2^(-1/3)
      computeGaussianKernel<kernel_size>(0.9732939207), // sqrt(1.6^2-1.2699208416^2) = 0.9732939207
      computeGaussianKernel<kernel_size>(1.2262734985), // sqrt(2.01587367983^2-1.6^2) = 1.2262734985
      computeGaussianKernel<kernel_size>(1.5450077936), // sqrt(2.5398416831^2-2.01587367983^2) = 1.5450077936
      computeGaussianKernel<kernel_size>(1.9465878415), // sqrt(3.2^2-2.5398416831^2) = 1.9465878415
      computeGaussianKernel<kernel_size>(2.4525469969)  // sqrt(4.03174735966^2-3.2^2) = 2.4525469969
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