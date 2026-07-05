#ifndef LAR_IO_IMAGE_IO_H
#define LAR_IO_IMAGE_IO_H

#include <string>
#include <opencv2/core.hpp>

// Minimal image I/O for LAR, backed by vendored stb (JPEG) + a small PFM codec.
//
// This is the ONLY translation unit that pulls in an image codec, and it lives in
// the `lar_io` leaf target. Only modules that actually read/write images (mapping,
// processing) link it; core and tracking never do, so a tracking-only binary carries
// no codec code. Interfaces still speak cv::Mat for now (that stays until the wider
// cv::Mat -> lar type migration); this only swaps the codec backend, not the types.

namespace lar {
namespace io {

  // Read an 8-bit grayscale JPEG into a CV_8UC1 Mat (matches cv::imread(..., IMREAD_GRAYSCALE)).
  // Returns an empty cv::Mat on failure, matching cv::imread's contract (callers check .empty()).
  cv::Mat imreadGray(const std::string& path);

  // Read an 8-bit color JPEG into a CV_8UC3 Mat in BGR order (matches cv::imread(..., IMREAD_COLOR)).
  // Returns an empty cv::Mat on failure, matching cv::imread's contract (callers check .empty()).
  cv::Mat imreadColor(const std::string& path);

  // Write a Mat as a baseline JPEG. Accepts CV_8UC1, CV_8UC3 (BGR) or CV_8UC4 (BGRA),
  // matching OpenCV's channel convention; BGR(A) is converted to RGB for encoding.
  // quality is 1..100. Throws std::runtime_error on failure.
  void imwriteJpeg(const std::string& path, const cv::Mat& image, int quality = 95);

  // Read a Portable Float Map into CV_32FC1 (1-channel "Pf") or CV_32FC3 (3-channel "PF").
  // Byte layout is OpenCV/standard PFM compatible (bottom-row-first, scale sign = endianness).
  // Returns an empty cv::Mat on failure, matching cv::imread's contract (callers check .empty()).
  cv::Mat imreadPFM(const std::string& path);

  // Write a CV_32FC1 or CV_32FC3 Mat as a little-endian PFM, OpenCV-compatible layout.
  // Non-float inputs are converted to float first. Throws std::runtime_error on failure.
  void imwritePFM(const std::string& path, const cv::Mat& image);

}  // namespace io
}  // namespace lar

#endif /* LAR_IO_IMAGE_IO_H */
