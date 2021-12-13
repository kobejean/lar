#include <stdint.h>
#include <iostream>

#include "geoar/process/bundle_adjustment.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

G2O_USE_OPTIMIZATION_LIBRARY(eigen);
G2O_USE_OPTIMIZATION_LIBRARY(dense);

namespace g2o {
G2O_REGISTER_TYPE_GROUP(expmap);
G2O_REGISTER_TYPE(PARAMS_CAMERAPARAMETERS, CameraParameters);
G2O_REGISTER_TYPE(VERTEX_SE3:EXPMAP, VertexSE3Expmap);
G2O_REGISTER_TYPE(EDGE_SE3:EXPMAP, EdgeSE3Expmap);
G2O_REGISTER_TYPE(EDGE_PROJECT_XYZ2UV:EXPMAP, EdgeProjectXYZ2UV);

G2O_REGISTER_TYPE_GROUP(slam3d);
G2O_REGISTER_TYPE(VERTEX_TRACKXYZ, VertexPointXYZ);
}

namespace geoar {

  BundleAdjustment::BundleAdjustment(MapProcessingData &data) {
    this->data = &data;
    optimizer.setVerbose(true);
    string solver_name = "lm_fix6_3";
    g2o::OptimizationAlgorithmProperty solver_property;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solver_name, solver_property);
    optimizer.setAlgorithm(algorithm);
  }

  void BundleAdjustment::construct(std::string directory) {

    size_t landmark_count = data->map.landmarkDatabase.landmarks.size();
    for (size_t i = 0; i < landmark_count; i++) {
      Landmark &landmark = data->map.landmarkDatabase.landmarks[i];
      if (landmark.sightings >= 3) {
        // Create feature point vertex
        g2o::VertexPointXYZ * vertex = new g2o::VertexPointXYZ();
        vertex->setId(i);
        // vertex->setFixed(true);
        vertex->setMarginalized(true);
        vertex->setEstimate(landmark.position);
        optimizer.addVertex(vertex);
      }
    }

    int frame_id = landmark_count;
    for (size_t i = 0; i < data->frames.size(); i++) {
      Frame const& frame = data->frames[i];


      g2o::VertexSE3Expmap * vertex = new g2o::VertexSE3Expmap();
      vertex->setId(frame_id);
      vertex->setEstimate(frame.pose);
      vertex->setFixed(frame_id == landmark_count);
      optimizer.addVertex(vertex);

      if (frame_id > landmark_count) {
        g2o::VertexSE3Expmap* v1 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id-1));
        g2o::VertexSE3Expmap* v2 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id));
        g2o::SE3Quat m = v2->estimate() * v1->estimate().inverse();

        g2o::EdgeSE3Expmap * e = new g2o::EdgeSE3Expmap();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setMeasurement(m);
        e->information() = Eigen::MatrixXd::Identity(6,6) * 100000000;
        optimizer.addEdge(e);
      }

      // Get camera intrinsics
      double focal_length = frame.intrinsics["focalLength"];
      Vector2d principle_point(frame.intrinsics["principlePoint"]["x"], frame.intrinsics["principlePoint"]["y"]);
      auto * cam_params = new g2o::CameraParameters(focal_length, principle_point, 0.);
      cam_params->setId(i+1);
      if (!optimizer.addParameter(cam_params)) {
        assert(false);
      }

      for (size_t j = 0; j < frame.landmarks.size(); j++) {
        
        size_t landmark_idx = frame.landmarks[j];
        Landmark &landmark = data->map.landmarkDatabase.landmarks[landmark_idx];
        if (landmark.sightings >= 3) {
          cv::KeyPoint keypoint = frame.kpts[j];
          Vector2d kp = Vector2d(keypoint.pt.x, keypoint.pt.y);

          g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(landmark_idx)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_id)));
          e->setMeasurement(kp);
          e->information() = Matrix2d::Identity();
          e->setParameterId(0, i+1);
          // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          // rk->setDelta(100.0);
          // e->setRobustKernel(rk);
          optimizer.addEdge(e);
        }
      }

      frame_id++;
    }
    
    printStats();
  }

  // Private methods

  void BundleAdjustment::printStats() {
    int count = 0;
    for (size_t i = 0; i < data->map.landmarkDatabase.landmarks.size(); i++) {
      if (data->map.landmarkDatabase.landmarks[i].sightings >= 3) {
        count++;
      }
    }
    cout << "usable landmarks: " << count << endl;

    for (Frame const& frame : data->frames) {
      // Projection projection(frame.frame_data);
      int count = 0;
      for (size_t i = 0; i < frame.landmarks.size(); i++) {
        size_t landmark_idx = frame.landmarks[i];
        Landmark &landmark = data->map.landmarkDatabase.landmarks[landmark_idx];
        if (landmark.sightings >= 3) {
          count++;
        }

        // cv::Point2f pt = projection.projectToImage(landmark.position);
        // cv::Point2f diff = frame.kpts[i].pt - pt;
        // float dist = sqrt(diff.dot(diff));
        // float ang_diff = angleDifference(frame.kpts[i].angle, landmark.kpt.angle);
        // if (dist > 200.0f || ang_diff > 30.0f) {
        //   cout << "diff: " << diff << endl;
        //   cout << "dist: " << dist << " ang_diff: " << ang_diff << endl;
        // }
      }
      cout << "frame landmarks: " << frame.landmarks.size() << endl;
      cout << "frame usable landmarks: " << count << endl;
    }
  }

  float BundleAdjustment::angleDifference(float alpha, float beta) {
    float phi = fmod(abs(beta - alpha), 360.f);
    return phi > 180.f ? 360.f - phi : phi;
  }
}