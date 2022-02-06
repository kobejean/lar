#ifndef LAR_CORE_LANDMARK_H
#define LAR_CORE_LANDMARK_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include "lar/core/utils/base64.h"
#include "lar/core/utils/json.h"

namespace lar {

  class Landmark {
    public:
      size_t id;
      Eigen::Vector3d position;
      Eigen::Vector3f orientation;
      cv::Mat desc;

      // For r-tree indexing
      Eigen::Vector2d index_center;
      double index_radius;

      Landmark();
      Landmark(const Eigen::Vector3d& position, const cv::Mat& desc, size_t id);

      static void concatDescriptions(const std::vector<Landmark>& landmarks, cv::Mat &desc);

#ifndef LAR_COMPACT_BUILD

      struct Observation {
        size_t landmark_id;
        size_t frame_id;
        long long timestamp;
        Eigen::Vector3d cam_position;
        cv::KeyPoint kpt;
        float depth;
        float depth_confidence;
        Eigen::Vector3f surface_normal;
      };
      // Auxilary data
      int sightings{0};
      long long last_seen;
      
      void recordObservation(Observation observation);
      bool isUseable() const;

#endif

  };

  static void to_json(nlohmann::json& j, const Landmark& l) {
    std::string desc64 = base64::base64_encode(l.desc);

    j = nlohmann::json{
      {"id", l.id},
      {"desc", desc64},
      {"position", l.position},
      {"orientation", l.orientation}
    };
  }

  static void from_json(const nlohmann::json& j, Landmark& l) {
    std::string desc64 = j.at("desc").get<std::string>();

    j.at("id").get_to(l.id);
    l.desc = base64::base64_decode(desc64, 1, 61, CV_8UC1);
    j.at("position").get_to(l.position);
    j.at("orientation").get_to(l.orientation);
  }

}

#endif /* LAR_CORE_LANDMARK_H */