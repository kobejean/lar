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
      cv::Mat desc;
      int sightings{0};
      long long last_seen;

      // For r-tree indexing
      Eigen::Vector2d index_center;
      double index_radius;

      Landmark();
      Landmark(Eigen::Vector3d &position, cv::Mat desc, size_t id);
      void recordSighting(Eigen::Vector3d &cam_position, long long timestamp) ;
      bool isUseable() const;

      static void concatDescriptions(std::vector<Landmark> landmarks, cv::Mat &desc);
  };
}

#endif /* LAR_CORE_LANDMARK_H */