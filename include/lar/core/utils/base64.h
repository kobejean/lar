#ifndef LAR_CORE_UTILS_BASE64_H
#define LAR_CORE_UTILS_BASE64_H

#include <iostream>
#include <opencv2/features2d.hpp>

namespace lar {
  namespace base64 {
    
    static const std::string base64_chars =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/";


    static inline bool is_base64(uchar c) {
      return (isalnum(c) || (c == '+') || (c == '/'));
    }

    static std::string base64_encode(uchar const* buf, unsigned int bufLen) {
      std::string ret;
      int i = 0;
      int j = 0;
      uchar char_array_3[3];
      uchar char_array_4[4];

      while (bufLen--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
          char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
          char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
          char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
          char_array_4[3] = char_array_3[2] & 0x3f;

          for(i = 0; (i <4) ; i++)
            ret += base64_chars[char_array_4[i]];
          i = 0;
        }
      }

      if (i)
      {
        for(j = i; j < 3; j++)
          char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
          ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
          ret += '=';
      }

      return ret;
    }

    // Decode into provided buffer, returns number of bytes written
    static size_t base64_decode(std::string const& encoded_string, uchar* output_buffer) {
      int in_len = static_cast<int>(encoded_string.size());
      int i = 0;
      int j = 0;
      int in_ = 0;
      uchar char_array_4[4], char_array_3[3];
      size_t out_pos = 0;

      while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
          for (i = 0; i < 4; i++)
            char_array_4[i] = static_cast<uchar>(base64_chars.find(char_array_4[i]));

          char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
          char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
          char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

          for (i = 0; (i < 3); i++)
            output_buffer[out_pos++] = char_array_3[i];
          i = 0;
        }
      }

      if (i) {
        for (j = i; j < 4; j++)
          char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
          char_array_4[j] = static_cast<uchar>(base64_chars.find(char_array_4[j]));

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++)
          output_buffer[out_pos++] = char_array_3[j];
      }

      return out_pos;
    }

    static std::string base64_encode(const cv::Mat& mat) {
      std::vector<uchar> data;
      data.reserve(mat.rows * mat.cols);

      for (int i = 0; i < mat.rows; i++) {
        const uchar *segment_start = mat.ptr(i);
        data.insert(data.end(), segment_start, segment_start + mat.cols * mat.elemSize());
      }
      return base64_encode(&data[0], data.size());
    }

    static cv::Mat base64_decode(std::string const& encoded_string, int rows, int cols, int type) {
      cv::Mat mat(rows, cols, type);
      base64_decode(encoded_string, mat.data);
      return mat;
    }

  }
}

#endif /* LAR_CORE_UTILS_BASE64_H */