#include "lar/io/image_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// stb owns exactly this translation unit. JPEG-only decode keeps the codec small.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_LINEAR
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace lar {
namespace io {

namespace {

  bool hostIsLittleEndian() {
    const uint32_t one = 1u;
    return *reinterpret_cast<const uint8_t*>(&one) == 1u;
  }

  void byteswap32(uint32_t& v) {
    v = ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |
        ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
  }

  // BGR(A) -> tightly packed RGB. `src` is CV_8UC3 or CV_8UC4. Returns w*h*3 bytes.
  std::vector<uint8_t> bgrToRgb(const cv::Mat& src) {
    const int w = src.cols, h = src.rows, ch = src.channels();
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * 3);
    size_t o = 0;
    for (int y = 0; y < h; ++y) {
      const uint8_t* row = src.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) {
        const uint8_t* px = row + static_cast<size_t>(x) * ch;
        out[o++] = px[2];  // R
        out[o++] = px[1];  // G
        out[o++] = px[0];  // B
      }
    }
    return out;
  }

}  // namespace

cv::Mat imreadGray(const std::string& path) {
  int w = 0, h = 0, channels_in_file = 0;
  // Force 1 channel: stb applies BT.601-style luma, matching cv::IMREAD_GRAYSCALE closely enough.
  uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels_in_file, 1);
  if (!data) {
    return cv::Mat();  // match cv::imread: empty Mat on failure, callers check .empty()
  }
  // Wrap-and-clone so the returned Mat owns its memory and we can free stb's buffer.
  cv::Mat wrapped(h, w, CV_8UC1, data);
  cv::Mat result = wrapped.clone();
  stbi_image_free(data);
  return result;
}

void imwriteJpeg(const std::string& path, const cv::Mat& image, int quality) {
  if (image.empty()) {
    throw std::runtime_error("lar::io::imwriteJpeg: empty image for '" + path + "'");
  }
  if (image.depth() != CV_8U) {
    throw std::runtime_error("lar::io::imwriteJpeg: expected 8-bit image for '" + path + "'");
  }

  const int w = image.cols, h = image.rows, ch = image.channels();
  int ok = 0;
  if (ch == 1) {
    cv::Mat cont = image.isContinuous() ? image : image.clone();
    ok = stbi_write_jpg(path.c_str(), w, h, 1, cont.data, quality);
  } else if (ch == 3 || ch == 4) {
    std::vector<uint8_t> rgb = bgrToRgb(image);  // OpenCV stores BGR(A); JPEG wants RGB
    ok = stbi_write_jpg(path.c_str(), w, h, 3, rgb.data(), quality);
  } else {
    throw std::runtime_error("lar::io::imwriteJpeg: unsupported channel count for '" + path + "'");
  }
  if (!ok) {
    throw std::runtime_error("lar::io::imwriteJpeg: failed to write '" + path + "'");
  }
}

cv::Mat imreadPFM(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return cv::Mat();  // match cv::imread: empty Mat on failure, callers check .empty()
  }

  std::string magic;
  int width = 0, height = 0;
  double scale = 0.0;
  f >> magic >> width >> height >> scale;
  if (!f || (magic != "Pf" && magic != "PF") || width <= 0 || height <= 0 || scale == 0.0) {
    return cv::Mat();  // malformed header -> empty, as cv::imread would
  }
  f.get();  // consume the single whitespace byte separating header from binary data

  const int channels = (magic == "PF") ? 3 : 1;
  const bool fileLittleEndian = scale < 0.0;
  const bool needSwap = fileLittleEndian != hostIsLittleEndian();

  cv::Mat out(height, width, channels == 3 ? CV_32FC3 : CV_32FC1);
  const size_t rowFloats = static_cast<size_t>(width) * channels;
  const std::streamsize rowBytes = static_cast<std::streamsize>(rowFloats * sizeof(float));

  // PFM scanlines run bottom-to-top: first row in the file is the bottom image row.
  for (int y = height - 1; y >= 0; --y) {
    f.read(reinterpret_cast<char*>(out.ptr<float>(y)), rowBytes);
    if (!f) {
      return cv::Mat();  // truncated data -> empty, as cv::imread would
    }
    if (needSwap) {
      uint32_t* p = out.ptr<uint32_t>(y);
      for (size_t i = 0; i < rowFloats; ++i) byteswap32(p[i]);
    }
  }

  const float inv = static_cast<float>(1.0 / std::abs(scale));
  if (inv != 1.0f) out *= inv;
  return out;
}

void imwritePFM(const std::string& path, const cv::Mat& image) {
  if (image.empty()) {
    throw std::runtime_error("lar::io::imwritePFM: empty image for '" + path + "'");
  }
  cv::Mat img = image;
  if (img.depth() != CV_32F) img.convertTo(img, CV_32F);
  const int channels = img.channels();
  if (channels != 1 && channels != 3) {
    throw std::runtime_error("lar::io::imwritePFM: expected 1 or 3 channel image for '" + path + "'");
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("lar::io::imwritePFM: cannot open '" + path + "' for writing");
  }

  const int w = img.cols, h = img.rows;
  // Header: magic, dimensions, scale (-1.0 => little-endian data, matching OpenCV on LE hosts).
  f << (channels == 3 ? "PF" : "Pf") << '\n' << w << ' ' << h << '\n' << "-1.0" << '\n';

  const size_t rowFloats = static_cast<size_t>(w) * channels;
  const std::streamsize rowBytes = static_cast<std::streamsize>(rowFloats * sizeof(float));
  std::vector<float> rgb;  // reused scratch for BGR->RGB when 3-channel

  // Write bottom-to-top to match PFM convention.
  for (int y = h - 1; y >= 0; --y) {
    const float* row = img.ptr<float>(y);
    if (channels == 3) {
      rgb.resize(rowFloats);
      for (int x = 0; x < w; ++x) {
        rgb[x * 3 + 0] = row[x * 3 + 2];  // R
        rgb[x * 3 + 1] = row[x * 3 + 1];  // G
        rgb[x * 3 + 2] = row[x * 3 + 0];  // B
      }
      f.write(reinterpret_cast<const char*>(rgb.data()), rowBytes);
    } else {
      f.write(reinterpret_cast<const char*>(row), rowBytes);
    }
  }
  if (!f) {
    throw std::runtime_error("lar::io::imwritePFM: write error on '" + path + "'");
  }
}

}  // namespace io
}  // namespace lar
