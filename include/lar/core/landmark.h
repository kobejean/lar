#ifndef LAR_CORE_LANDMARK_H
#define LAR_CORE_LANDMARK_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

namespace lar {

  class Landmark {
    public:
      size_t id;
      Eigen::Vector3d position;
      Eigen::Vector3f orientation;
      cv::Mat desc;
      long long last_seen;

      // For r-tree indexing
      Eigen::Vector2d index_center;
      double index_radius;

      Landmark();
      Landmark(const Eigen::Vector3d& position, const cv::Mat& desc, size_t id);

      static void concatDescriptions(const std::vector<Landmark>& landmarks, cv::Mat &desc);

#ifndef LAR_COMPACT_BUILD

      struct Observation {
        long long timestamp;
        Eigen::Vector3d cam_position;
        Eigen::Vector3f surface_normal;
      };
      // Auxilary data
      int sightings{0};
      std::vector<Observation> obs;
      
      void recordObservation(Observation observation);
      bool isUseable() const;

#endif

  };
}

#endif /* LAR_CORE_LANDMARK_H */