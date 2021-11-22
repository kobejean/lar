#include <iostream>

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <opencv2/features2d.hpp>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class Frame {
    public: 
      static cv::Ptr<cv::AKAZE> detector;

      json transform;
      g2o::SE3Quat pose;

      Frame(json& frame_data, std::string directory);

    private:

      void createPose(json& t);
      void createFeatures(int id, std::string directory);
  };
}