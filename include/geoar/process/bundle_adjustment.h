#ifndef GEOAR_BUNDLE_ADJUSTMENT_H
#define GEOAR_BUNDLE_ADJUSTMENT_H

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o/core/sparse_optimizer.h"

#include "geoar/core/landmark.h"
#include "geoar/core/frame.h"
#include "geoar/process/map_processing_data.h"

namespace geoar {

  class BundleAdjustment {
    public:
      g2o::SparseOptimizer optimizer;
      MapProcessingData* data;

      BundleAdjustment(MapProcessingData &data);
      void construct();

    private:
      struct Stats {
        size_t total_usable_landmarks = 0;
        std::vector<size_t> landmarks;
        std::vector<size_t> usable_landmarks;

        void print();
      };
      Stats _stats;
      
      bool addLandmark(Landmark const &landmark, size_t id);
      void addPose(g2o::SE3Quat const &pose, size_t id, bool fixed);
      void addOdometry(size_t last_frame_id);
      void addIntrinsics(nlohmann::json const &intrinsics, size_t id);
      void addLandmarkMeasurements(Frame const &frame, size_t frame_id, size_t params_id);
  };

}

#endif /* GEOAR_BUNDLE_ADJUSTMENT_H */