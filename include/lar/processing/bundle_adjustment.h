#ifndef LAR_PROCESSING_BUNDLE_ADJUSTMENT_H
#define LAR_PROCESSING_BUNDLE_ADJUSTMENT_H

#include <Eigen/Core>

#include "g2o/core/sparse_optimizer.h"

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
      void optimize();

    private:
      struct Stats {
        size_t total_usable_landmarks = 0;
        std::vector<size_t> landmarks;
        std::vector<size_t> usable_landmarks;

        void print();
      };
      Stats _stats;
      
      bool addLandmark(const Landmark& landmark, size_t id);
      void addPose(const Eigen::Matrix4d& extrinsics, size_t id, bool fixed);
      void addOdometry(size_t last_frame_id);
      void addIntrinsics(const Eigen::Matrix3d& intrinsics, size_t id);
      void addLandmarkMeasurements(const Landmark& landmark, size_t id);

      void updateLandmark(size_t landmark_id);
  };

}

#endif /* LAR_PROCESSING_BUNDLE_ADJUSTMENT_H */