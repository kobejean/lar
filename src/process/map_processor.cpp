
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/process/map_processor.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  MapProcessor::MapProcessor() {
    optimizer.setVerbose(true);
    string solverName = "lm_fix6_3";
  #ifdef G2O_HAVE_CHOLMOD
    solverName = "lm_fix6_3_cholmod";
  #else
    solverName = "lm_fix6_3";
  #endif
    g2o::OptimizationAlgorithmProperty solverProperty;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty);
    optimizer.setAlgorithm(algorithm);
  }

  void MapProcessor::parseMap(std::ifstream& map_ifs) {
    json map = json::parse(map_ifs);
    json point_cloud = map["pointCloud"];
    json camera_points = point_cloud["cameraPoints"];
    json feature_points = point_cloud["featurePoints"];
    // json location_points = point_cloud["locationPoints"];

    parseVertices(feature_points, camera_points);
    addFeaturePoints(feature_points);
    addCameraPoints(camera_points);
  }

  // Private functions

  void MapProcessor::parseVertices(json& feature_points, json& camera_points) {
    int vertex_id = 0;

    // Populate `points[]`
    for (auto& el : feature_points.items()) {
      vertex_id++;
      json fp = el.value();
      string fp_uuid = fp["identifier"];
      json t = fp["transform"];
      Vector3d position(t[3][0], t[3][1], t[3][2]);
      points[fp_uuid] = { vertex_id, fp_uuid, position };
    }

    // Populate `poses[]`
    for (auto& el : camera_points.items()) {
      vertex_id++;
      json cp = el.value();
      string cp_uuid = cp["identifier"];

      // Create transform
      json t = cp["transform"];
      Matrix3d rot;
      rot << t[0][0], t[1][0], t[2][0],
             t[0][1], t[1][1], t[2][1],
             t[0][2], t[1][2], t[2][2]; 
      Vector3d position(t[3][0], t[3][1], t[3][2]);
      Quaterniond orientation(rot);
      g2o::SE3Quat transform = g2o::SE3Quat(orientation, position).inverse();

      poses[cp_uuid] = { vertex_id, cp_uuid, transform };

      // Count feature point observations `points[].obs_count`
      for (auto& el : cp["featurePoints"].items()) {
        json fp = el.value();
        std::string fp_uuid = fp["identifier"];
        if (points.count(fp_uuid)) {
          points[fp_uuid].obs_count++;
        }
      }
    }
  }

  void MapProcessor::addFeaturePoints(json& feature_points) {
    for (auto& el : feature_points.items()) {
      json fp = el.value();
      auto fp_uuid = fp["identifier"];
      Point point = points[fp_uuid];

      if (point.obs_count > 2) {
        // Create feature point vertex
        g2o::VertexPointXYZ * vertex = new g2o::VertexPointXYZ();
        vertex->setId(point.id);
        vertex->setMarginalized(true);
        vertex->setEstimate(point.position);
        optimizer.addVertex(vertex);
      }
    }
  }

  void MapProcessor::addCameraPoints(json& camera_points) {
    int cp_count = 0;
    int last_cp_id = 0;

    for (auto& el : camera_points.items()) {
      json cp = el.value();
      auto cp_uuid = cp["identifier"];

      // Handle camera points that have associated feature points
      if (cp["featurePoints"].size() > 0) {
        // Create camera point vertex
        Pose pose = poses[cp_uuid];
        int cp_id = pose.id;
        g2o::VertexSE3Expmap * vertex = new g2o::VertexSE3Expmap();
        vertex->setId(cp_id);
        vertex->setEstimate(pose.transform);
        if (cp_count == 0){
          vertex->setFixed(true); // Fix the first camera point
        }
        optimizer.addVertex(vertex);

        // Add edge from camera point to camera point after the second camera point has been added
        if (cp_count > 0) {
          g2o::EdgeSE3Expmap * e = new g2o::EdgeSE3Expmap();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(last_cp_id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(cp_id)));
          e->information() = Eigen::MatrixXd::Identity(6,6);
          optimizer.addEdge(e);
        }
        last_cp_id = cp_id;
        cp_count++;

        // Get camera intrinsics
        json intrinsics = cp["intrinsics"];
        double focal_length = intrinsics["focalLength"];
        Vector2d principle_point(intrinsics["principlePoint"]["x"], intrinsics["principlePoint"]["y"]);
        auto * cam_params = new g2o::CameraParameters(focal_length, principle_point, 0.);
        cam_params->setId(cp_count);
        if (!optimizer.addParameter(cam_params)) {
          assert(false);
        }

        // Add edges connecting from camera pose to feature point position
        for (auto& el : cp["featurePoints"].items()) {
          json fp = el.value();
          std::string fp_uuid = fp["identifier"];
          Point point = points[fp_uuid];
          int fp_id = point.id;
          double kp_x = fp["keyPoint"]["x"];
          double kp_y = fp["keyPoint"]["y"];
          Vector2d kp = Vector2d(principle_point[0]*2 - kp_x, kp_y);
          Vector2d z = cam_params->cam_map(pose.transform.map(point.position));
          Vector2d diff = z - kp;

          // TODO: Revisit fp_id > 0 
          if (point.obs_count > 2 && fp_id > 0 && abs(diff[0]) < 100 && abs(diff[1]) < 100) {
            g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(fp_id)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(cp_id)));
            e->setMeasurement(kp);
            e->information() = Matrix2d::Identity();
            e->setParameterId(0, cp_count);
            // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            // rk->setDelta(100.0);
            // e->setRobustKernel(rk);
            optimizer.addEdge(e);
          }
        }
      }
    }
  }
}