#ifndef LAR_CORE_LANDMARK_H
#define LAR_CORE_LANDMARK_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include "lar/core/utils/base64.h"
#include "lar/core/utils/json.h"
#include "lar/core/spacial/rect.h"

namespace lar {

  class Landmark {
    public:
      size_t id;
      Eigen::Vector3d position;
      Eigen::Vector3f orientation;
      cv::Mat desc;
      Rect bounds;

      Landmark();
      Landmark(const Eigen::Vector3d& position, const cv::Mat& desc, size_t id);

      static cv::Mat concatDescriptions(const std::vector<Landmark>& landmarks);

#ifndef LAR_COMPACT_BUILD

      struct Observation {
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
      std::vector<Observation> obs;
      long long last_seen;
      bool is_matched{false};
      
      void recordObservation(Observation observation);
      bool isUseable() const;

#endif

  };

  static void to_json(nlohmann::json& j, const Landmark& l) {
    cv::Mat desc;
    l.desc.convertTo(desc, CV_8U);
    std::string desc64 = base64::base64_encode(desc);

    j = nlohmann::json{
      {"id", l.id},
      {"desc", desc64},
      {"position", l.position},
      {"orientation", l.orientation},
      {"bounds", l.bounds},
      {"sightings", l.sightings}
    };
  }

  static void from_json(const nlohmann::json& j, Landmark& l) {
    std::string desc64 = j.at("desc").get<std::string>();
    cv::Mat desc = base64::base64_decode(desc64, 1, -1, CV_8U);
    // desc.convertTo(desc, CV_32F);

    j.at("id").get_to(l.id);
    l.desc = desc;
    j.at("position").get_to(l.position);
    j.at("orientation").get_to(l.orientation);
    j.at("bounds").get_to(l.bounds);
    j.at("sightings").get_to(l.sightings);
  }

}

#endif /* LAR_CORE_LANDMARK_H */