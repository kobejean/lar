// Benchmark: CPU vs GPU Gaussian pyramid construction
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>

namespace lar {

void benchmarkCPUPyramid(const cv::Mat& base, int iterations) {
    using Clock = std::chrono::high_resolution_clock;
    std::vector<double> times;

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<cv::Mat> pyr(24); // 4 octaves × 6 levels
        std::vector<double> sigmas = {1.6, 2.0, 2.5, 3.2, 4.0, 5.0};

        auto start = Clock::now();

        // Simulate SIFT pyramid construction
        for (int o = 0; o < 4; o++) {
            cv::Mat octaveBase;
            if (o == 0) {
                octaveBase = base;
            } else {
                cv::resize(pyr[(o-1)*6 + 3], octaveBase,
                          cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            }

            pyr[o*6] = octaveBase.clone();
            for (int i = 1; i < 6; i++) {
                cv::GaussianBlur(pyr[o*6 + i-1], pyr[o*6 + i],
                                cv::Size(), sigmas[i], sigmas[i]);
            }
        }

        auto end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // Convert to ms
    }

    // Calculate statistics
    double sum = 0, min = times[0], max = times[0];
    for (double t : times) {
        sum += t;
        min = std::min(min, t);
        max = std::max(max, t);
    }
    double avg = sum / times.size();

    std::cout << "CPU Gaussian Pyramid (" << iterations << " iterations):\n"
              << "  Average: " << avg << " ms\n"
              << "  Min: " << min << " ms\n"
              << "  Max: " << max << " ms\n"
              << "  Image: " << base.cols << "x" << base.rows << "\n";
}

// Compare different image sizes to understand GPU overhead scaling
void benchmarkSizes() {
    std::cout << "\n=== CPU Gaussian Pyramid Benchmark ===\n\n";

    // Small image (typical mobile AR frame)
    cv::Mat small(480, 640, CV_32F);
    cv::randu(small, 0, 255);
    std::cout << "Small image (640x480):\n";
    benchmarkCPUPyramid(small, 100);

    // Medium image (upscaled for SIFT)
    cv::Mat medium(960, 1280, CV_32F);
    cv::randu(medium, 0, 255);
    std::cout << "\nMedium image (1280x960):\n";
    benchmarkCPUPyramid(medium, 50);

    // Large image
    cv::Mat large(1920, 2560, CV_32F);
    cv::randu(large, 0, 255);
    std::cout << "\nLarge image (2560x1920):\n";
    benchmarkCPUPyramid(large, 20);

    std::cout << "\n=== Key Insight ===\n"
              << "If CPU time > 5ms per image AND processing multiple images,\n"
              << "then GPU batching will likely help. Otherwise, CPU+SIMD wins.\n";
}

} // namespace lar

// Uncomment to run standalone benchmark:
// int main() {
//     lar::benchmarkSizes();
//     return 0;
// }