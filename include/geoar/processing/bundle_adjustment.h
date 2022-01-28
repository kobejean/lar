#ifndef GEOAR_PROCESSING_BUNDLE_ADJUSTMENT_H
#define GEOAR_PROCESSING_BUNDLE_ADJUSTMENT_H

#include <Eigen/Core>

#include "g2o/core/sparse_optimizer.h"

#include "geoar/core/landmark.h"
#include "geoar/mapping/frame.h"
#include "geoar/mapping/mapper.h"

namespace geoar {

  class BundleAdjustment {
    public:
      g2o::SparseOptimizer optimizer;
      Mapper::Data* data;

      BundleAdjustment(Mapper::Data &data);
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
      
      bool addLandmark(Landmark const &landmark, size_t id);
      void addPose(Eigen::Matrix4d const &extrinsics, size_t id, bool fixed);
      void addOdometry(size_t last_frame_id);
      void addIntrinsics(Eigen::Matrix3d const &intrinsics, size_t id);
      void addLandmarkMeasurements(Frame const &frame, size_t frame_id, size_t params_id);

      void updateLandmark(size_t landmark_id);
  };

}

#endif /* GEOAR_PROCESSING_BUNDLE_ADJUSTMENT_H */