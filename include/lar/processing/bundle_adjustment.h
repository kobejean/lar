#ifndef LAR_PROCESSING_BUNDLE_ADJUSTMENT_H
#define LAR_PROCESSING_BUNDLE_ADJUSTMENT_H

#include <Eigen/Core>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/se3quat.h"

#include "lar/core/landmark.h"
#include "lar/mapping/frame.h"
#include "lar/mapping/mapper.h"

namespace lar {

  class BundleAdjustment {
    public:
      g2o::SparseOptimizer optimizer;
      std::shared_ptr<Mapper::Data> data;

      BundleAdjustment(std::shared_ptr<Mapper::Data> data);
      void construct();
      void reset();
      void optimize();
      void update();

    private:
      struct Stats {
        size_t total_usable_landmarks = 0;
        std::vector<size_t> landmarks;
        std::vector<size_t> usable_landmarks;

        void print();
      };
      Stats _stats;
      std::vector<g2o::EdgeProjectXYZ2UVD*> _landmark_edges;
      
      bool addLandmark(const Landmark &landmark, size_t id);
      void addPose(const Eigen::Matrix4d& extrinsics, size_t id, bool fixed);
      void addGravityConstraint(size_t frame_id);
      void addOdometry(size_t frame_id);
      void addIntrinsics(const Eigen::Matrix3d& intrinsics, size_t id);
      void addLandmarkMeasurements(const Landmark& landmark, size_t id);

      void markOutliers(double chi_threshold);

      void updateLandmarks();
      void updateAnchors();

      static g2o::SE3Quat poseFromExtrinsics(const Eigen::Matrix4d& extrinsics);
      static Eigen::Matrix4d extrinsicsFromPose(const g2o::SE3Quat& pose);
  };

}

#endif /* LAR_PROCESSING_BUNDLE_ADJUSTMENT_H */
