#ifndef GEOAR_MAP_PROCESSING_DATA_H
#define GEOAR_MAP_PROCESSING_DATA_H

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/core/frame.h"
#include "geoar/core/map.h"
#include "geoar/core/landmark.h"
#include "geoar/process/projection.h"
#include "geoar/process/vision.h"

using namespace std;

namespace geoar {

  class MapProcessingData {
    public:
      Vision vision;
      Map map;
      vector<Frame> frames;
      cv::Mat desc;

      MapProcessingData();
      void loadRawData(string directory);
      void loadFrameData(json& frame_data, std::string directory);

    private:
      vector<size_t> getLandmarks(vector<cv::KeyPoint> &kpts, cv::Mat &desc, vector<float> &depth, json& transform);
      vector<float> getDepthValues(vector<cv::KeyPoint> &kpts, std::string depth_filepath, cv::Size img_size);
      std::map<size_t, size_t> getMatches(cv::Mat &desc);
      std::string getPathPrefix(int id, std::string directory);
  };

}

#endif /* GEOAR_MAP_PROCESSING_DATA_H */