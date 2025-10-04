#ifndef LAR_CORE_LANDMARK_H
#define LAR_CORE_LANDMARK_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include "lar/core/utils/base64.h"
#include "lar/core/utils/json.h"
#include "lar/core/spatial/rect.h"

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

      static cv::Mat concatDescriptions(const std::vector<Landmark*>& landmarks);
      
      /**
       * Calculates spatial bounds for a landmark based on observing camera positions.
       * The bounds encompass both the landmark position and all camera positions that observed it,
       * with a 10% margin based on the maximum distance.
       * 
       * @param landmark_position Position of the landmark in 3D space
       * @param camera_positions Positions of cameras that observed this landmark
       * @return Rectangular bounds in XZ plane (Y is up in ARKit convention)
       */
      static Rect calculateSpatialBounds(const Eigen::Vector3d& landmark_position,
                                        const std::vector<Eigen::Vector3d>& camera_positions,
                                        double marginRatio = 0.2);
      
      /**
       * Updates this landmark's bounds based on observing camera positions.
       * Convenience method that calls calculateSpatialBounds and updates the bounds field.
       *
       * @param camera_positions Positions of cameras that observed this landmark
       * @param marginRatio Ratio of max distance to use as margin (default: 0.2)
       */
      void updateBounds(const std::vector<Eigen::Vector3d>& camera_positions, double marginRatio = 0.2);

#ifndef LAR_COMPACT_BUILD
      /**
       * Updates this landmark's bounds based on its stored observations.
       * Extracts camera positions from the observation data and updates bounds.
       *
       * @param marginRatio Ratio of max distance to use as margin (default: 0.2)
       */
      void updateBoundsFromObservations(double marginRatio = 0.2);

      struct Observation {
        size_t frame_id;
        long long timestamp;
        Eigen::Matrix4d cam_pose;
        cv::KeyPoint kpt;
        float depth;
        float depth_confidence;
        Eigen::Vector3f surface_normal;
      };
      // Auxilary data
      int sightings{0};
      std::vector<Observation> obs;
      long long last_seen{-1};
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
    const std::string& desc64 = j.at("desc").get_ref<const std::string&>();
    l.desc = base64::base64_decode(desc64, 1, -1, CV_8U);

    j.at("id").get_to(l.id);
    j.at("position").get_to(l.position);
    j.at("orientation").get_to(l.orientation);
    j.at("bounds").get_to(l.bounds);
    j.at("sightings").get_to(l.sightings);

    #ifndef LAR_COMPACT_BUILD
    // Initialize members not in JSON to default values
    l.last_seen = -1;
    l.is_matched = false;
    l.obs.clear(); // Clear any existing observations
    #endif
  }

}

#endif /* LAR_CORE_LANDMARK_H */