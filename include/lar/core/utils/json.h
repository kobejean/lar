#ifndef LAR_CORE_UTILS_JSON_H
#define LAR_CORE_UTILS_JSON_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

#include "lar/core/utils/base64.h"
#include "lar/core/map.h"

namespace lar {

  static void to_json(nlohmann::json& j, const Landmark& l) {
    std::string desc64 = base64_encode(l.desc);

    j = nlohmann::json{ {"id", l.id}, {"desc", desc64}, {"position", l.position}, };
  }

  static void from_json(const nlohmann::json& j, Landmark& l) {
    std::string desc64 = j.at("desc").get<std::string>();

    j.at("id").get_to(l.id);
    l.desc = base64_decode(desc64, 1, 61, CV_8UC1);
    j.at("position").get_to(l.position);
  }

  static void to_json(nlohmann::json& j, const Map& m) {
    j = nlohmann::json{ {"landmarks", m.landmarks.all}, };
  }

  static void from_json(const nlohmann::json& j, Map& m) {
    std::vector<Landmark> landmarks = j.at("landmarks").get<std::vector<Landmark>>();
    m.landmarks.insert(landmarks);
  }

}


namespace Eigen {

  template< typename Scalar_, int Rows_, int Cols_ >
  static void to_json( nlohmann::json& j, const Matrix< Scalar_, Rows_, Cols_ >& mat )
  {
    j = std::vector<Scalar_>(mat.data(), mat.data() + mat.size());
  }

  template< typename Scalar_, int Rows_, int Cols_ >
  static void from_json( const nlohmann::json& j, Matrix< Scalar_, Rows_, Cols_ >& mat )
  {
    mat = Eigen::Matrix<Scalar_, Rows_, Cols_>(j.get<std::vector<Scalar_>>().data());
  }
  
}

#endif /* LAR_CORE_UTILS_JSON_H */