#ifndef GEOAR_JSON_H
#define GEOAR_JSON_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

#include "geoar/core/utils/base64.h"
#include "geoar/core/map.h"

namespace geoar {

  void to_json(nlohmann::json& j, const Landmark& l) {
    std::string desc64 = base64_encode(l.desc);
    j = nlohmann::json{ {"id", l.id}, {"desc", desc64 }, {"position", l.position}, };
  }

  void from_json(const nlohmann::json& j, Landmark& l) {
    std::string desc64 = j.at("desc").get<std::string>();
    std::vector<double> position = j.at("position").get<std::vector<double>>();

    j.at("id").get_to(l.id);
    l.desc = base64_decode(desc64, 1, 61, CV_8UC1);
    l.position = Eigen::Vector3d(position.data());
  }

  void to_json(nlohmann::json& j, const Map& m) {
    std::vector<Landmark> landmarks;
    for (size_t i = 0; i < m.landmarks.size(); i++) {
      landmarks.push_back(m.landmarks[i]);
    }
    
    j = nlohmann::json{ {"landmarks", nlohmann::json(landmarks)}, };
  }

  void from_json(const nlohmann::json& j, Map& m) {
    std::vector<Landmark> landmarks = j.at("landmarks").get<std::vector<Landmark>>();

    m.landmarks.insert(landmarks);
  }

}

namespace nlohmann {

  // Eigen::Matrix
  template <typename _Scalar, int _Rows, int _Cols>
  struct adl_serializer<Eigen::Matrix<_Scalar, _Rows, _Cols>> {
    static void to_json(json& j, Eigen::Matrix<_Scalar, _Rows, _Cols>& mat) {
      j = mat.data;
    }

    static Eigen::Matrix<_Scalar, _Rows, _Cols> from_json(const json& j) {
      return { j.get<std::vector<_Scalar>>().data() };
    }
  };

}

#endif /* GEOAR_JSON_H */