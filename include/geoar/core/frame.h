#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class Frame {
    public:
      json transform;
      g2o::SE3Quat pose;

      Frame(json& frame_data, std::string directory);

    private:

      void createPose(json& t);
  };
}