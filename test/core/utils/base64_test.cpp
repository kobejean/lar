#include <gtest/gtest.h>
#include "lar/core/utils/base64.h"
#include <vector>
#include <cstring>

using namespace lar::base64;

// ============================================================================
// Buffer-based decode API tests
// ============================================================================

TEST(Base64Test, DecodeToBuffer_BasicString) {
  // "Hello" encodes to "SGVsbG8="
  std::string encoded = "SGVsbG8=";
  uchar buffer[10] = {0};

  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 5);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer), bytes_written), "Hello");
}

TEST(Base64Test, DecodeToBuffer_ExactSize) {
  // "Test" encodes to "VGVzdA=="
  std::string encoded = "VGVzdA==";
  uchar buffer[4];  // Exact size needed

  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 4);
  EXPECT_EQ(0, std::memcmp(buffer, "Test", 4));
}

TEST(Base64Test, DecodeToBuffer_BufferTooSmall) {
  // "HelloWorld" encodes to "SGVsbG9Xb3JsZA=="
  std::string encoded = "SGVsbG9Xb3JsZA==";
  uchar buffer[5];  // Too small for 10 bytes

  // Should throw when buffer is exceeded
  EXPECT_THROW({
    base64_decode(encoded, buffer, sizeof(buffer));
  }, std::runtime_error);
}

TEST(Base64Test, DecodeToBuffer_EmptyString) {
  std::string encoded = "";
  uchar buffer[10] = {0};

  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 0);
}

TEST(Base64Test, DecodeToBuffer_SingleChar) {
  // "A" encodes to "QQ=="
  std::string encoded = "QQ==";
  uchar buffer[10] = {0};

  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 1);
  EXPECT_EQ(buffer[0], 'A');
}

TEST(Base64Test, DecodeToBuffer_TwoChars) {
  // "AB" encodes to "QUI="
  std::string encoded = "QUI=";
  uchar buffer[10] = {0};

  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 2);
  EXPECT_EQ(buffer[0], 'A');
  EXPECT_EQ(buffer[1], 'B');
}

TEST(Base64Test, DecodeToBuffer_BinaryData) {
  // Binary data: {0xFF, 0x00, 0xAB, 0xCD}
  uchar original[4] = {0xFF, 0x00, 0xAB, 0xCD};
  std::string encoded = base64_encode(original, 4);

  uchar buffer[10] = {0};
  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 4);
  EXPECT_EQ(buffer[0], 0xFF);
  EXPECT_EQ(buffer[1], 0x00);
  EXPECT_EQ(buffer[2], 0xAB);
  EXPECT_EQ(buffer[3], 0xCD);
}

TEST(Base64Test, DecodeToBuffer_MultipleOfThree) {
  // 6 bytes should encode/decode cleanly (no padding)
  uchar original[6] = {1, 2, 3, 4, 5, 6};
  std::string encoded = base64_encode(original, 6);

  uchar buffer[10] = {0};
  size_t bytes_written = base64_decode(encoded, buffer, sizeof(buffer));

  EXPECT_EQ(bytes_written, 6);
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(buffer[i], original[i]);
  }
}

TEST(Base64Test, DecodeToBuffer_LargeData) {
  // Test with larger dataset (100 bytes)
  std::vector<uchar> original(100);
  for (int i = 0; i < 100; i++) {
    original[i] = static_cast<uchar>(i);
  }

  std::string encoded = base64_encode(original.data(), original.size());

  std::vector<uchar> buffer(150);  // Larger than needed
  size_t bytes_written = base64_decode(encoded, buffer.data(), buffer.size());

  EXPECT_EQ(bytes_written, 100);
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(buffer[i], original[i]);
  }
}

TEST(Base64Test, DecodeToBuffer_UnlimitedSize) {
  // buffer_size = 0 means no limit (unsafe mode)
  std::string encoded = "SGVsbG9Xb3JsZA==";  // "HelloWorld"
  uchar buffer[20] = {0};

  size_t bytes_written = base64_decode(encoded, buffer, 0);  // No size check

  EXPECT_EQ(bytes_written, 10);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer), bytes_written), "HelloWorld");
}

TEST(Base64Test, DecodeToBuffer_OffByOne) {
  // Buffer is one byte too small
  std::string encoded = "QUJDRA==";  // "ABCD"
  uchar buffer[3];  // Need 4 bytes

  EXPECT_THROW({
    base64_decode(encoded, buffer, sizeof(buffer));
  }, std::runtime_error);
}

// ============================================================================
// Round-trip tests (encode -> decode -> verify)
// ============================================================================

TEST(Base64Test, RoundTrip_AllByteValues) {
  // Test all possible byte values
  std::vector<uchar> original(256);
  for (int i = 0; i < 256; i++) {
    original[i] = static_cast<uchar>(i);
  }

  std::string encoded = base64_encode(original.data(), original.size());

  std::vector<uchar> buffer(300);
  size_t bytes_written = base64_decode(encoded, buffer.data(), buffer.size());

  EXPECT_EQ(bytes_written, 256);
  for (int i = 0; i < 256; i++) {
    EXPECT_EQ(buffer[i], original[i]);
  }
}

TEST(Base64Test, RoundTrip_EdgeCaseLengths) {
  // Test lengths: 0, 1, 2, 3, 4, 5, 6 (different padding scenarios)
  for (size_t len = 0; len <= 6; len++) {
    std::vector<uchar> original(len);
    for (size_t i = 0; i < len; i++) {
      original[i] = static_cast<uchar>('A' + i);
    }

    std::string encoded = base64_encode(original.data(), original.size());

    std::vector<uchar> buffer(10);
    size_t bytes_written = base64_decode(encoded, buffer.data(), buffer.size());

    EXPECT_EQ(bytes_written, len);
    for (size_t i = 0; i < len; i++) {
      EXPECT_EQ(buffer[i], original[i]);
    }
  }
}

// ============================================================================
// OpenCV Mat integration tests
// ============================================================================

TEST(Base64Test, MatEncodeDecode_8UC1) {
  // Create a simple 3x3 grayscale matrix
  cv::Mat original(3, 3, CV_8UC1);
  for (int i = 0; i < 9; i++) {
    original.data[i] = static_cast<uchar>(i * 10);
  }

  std::string encoded = base64_encode(original);
  cv::Mat decoded = base64_decode(encoded, 3, 3, CV_8UC1);

  EXPECT_EQ(decoded.rows, 3);
  EXPECT_EQ(decoded.cols, 3);
  EXPECT_EQ(decoded.type(), CV_8UC1);

  for (int i = 0; i < 9; i++) {
    EXPECT_EQ(decoded.data[i], original.data[i]);
  }
}

TEST(Base64Test, MatEncodeDecode_8UC3) {
  // Create a 2x2 RGB matrix
  cv::Mat original(2, 2, CV_8UC3);
  for (int i = 0; i < 12; i++) {
    original.data[i] = static_cast<uchar>(i * 20);
  }

  std::string encoded = base64_encode(original);
  cv::Mat decoded = base64_decode(encoded, 2, 2, CV_8UC3);

  EXPECT_EQ(decoded.rows, 2);
  EXPECT_EQ(decoded.cols, 2);
  EXPECT_EQ(decoded.type(), CV_8UC3);

  for (int i = 0; i < 12; i++) {
    EXPECT_EQ(decoded.data[i], original.data[i]);
  }
}

TEST(Base64Test, MatEncodeDecode_SizeMismatch) {
  cv::Mat original(3, 3, CV_8UC1);
  for (int i = 0; i < 9; i++) {
    original.data[i] = static_cast<uchar>(i);
  }

  std::string encoded = base64_encode(original);

  // Try to decode with wrong dimensions
  EXPECT_THROW({
    cv::Mat decoded = base64_decode(encoded, 2, 2, CV_8UC1);  // Wrong size
  }, std::runtime_error);
}

TEST(Base64Test, MatEncodeDecode_32FC1) {
  // Test with floating point data
  cv::Mat original(2, 2, CV_32FC1);
  original.at<float>(0, 0) = 1.5f;
  original.at<float>(0, 1) = 2.5f;
  original.at<float>(1, 0) = 3.5f;
  original.at<float>(1, 1) = 4.5f;

  std::string encoded = base64_encode(original);
  cv::Mat decoded = base64_decode(encoded, 2, 2, CV_32FC1);

  EXPECT_EQ(decoded.type(), CV_32FC1);
  EXPECT_FLOAT_EQ(decoded.at<float>(0, 0), 1.5f);
  EXPECT_FLOAT_EQ(decoded.at<float>(0, 1), 2.5f);
  EXPECT_FLOAT_EQ(decoded.at<float>(1, 0), 3.5f);
  EXPECT_FLOAT_EQ(decoded.at<float>(1, 1), 4.5f);
}
