#include <stdint.h>
#include <iostream>

#include "geoar/process/graph_construction.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  GraphConstruction::GraphConstruction(MapProcessingData &data) {
    this->data = &data;
  }

  void GraphConstruction::processRawData(string directory) {
    ifstream metadata_ifs(directory + "/metadata.json");
    json metadata = json::parse(metadata_ifs);

    for (json frame_data : metadata["frames"]) {
      processFrameData(frame_data, directory);
    }
  }

  void GraphConstruction::processFrameData(json& frame_data, std::string directory) {
    int id = frame_data["id"];
    Frame frame(frame_data);

    // Create filename paths
    std::string path_prefix = getPathPrefix(id, directory);
    std::string img_filepath = path_prefix + "image.jpeg";
    std::string depth_filepath = path_prefix + "depth.pfm";

    // Load image
    cout << "loading: " << img_filepath << endl;
    cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);

    // Extract features
    cv::Mat desc;
    vision.extractFeatures(image, cv::noArray(), frame.kpts, desc);
    cout << "features: " << frame.kpts.size() << endl;

    frame.depth = getDepthValues(frame.kpts, depth_filepath, image.size());
    frame.landmarks = getLandmarks(frame.kpts, desc, frame.depth, frame_data);

    data->frames.push_back(frame);
  }

  void GraphConstruction::construct(std::string directory) {
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
        data->optimizer.addVertex(vertex);
      }
    }

    int frame_id = landmark_count;
    for (size_t i = 0; i < data->frames.size(); i++) {
      Frame const& frame = data->frames[i];


      g2o::VertexSE3Expmap * vertex = new g2o::VertexSE3Expmap();
      vertex->setId(frame_id);
      vertex->setEstimate(frame.pose);
      vertex->setFixed(frame_id == landmark_count);
      data->optimizer.addVertex(vertex);

      if (frame_id > landmark_count) {
        g2o::VertexSE3Expmap* v1 = dynamic_cast<g2o::VertexSE3Expmap*>(data->optimizer.vertex(frame_id-1));
        g2o::VertexSE3Expmap* v2 = dynamic_cast<g2o::VertexSE3Expmap*>(data->optimizer.vertex(frame_id));
        g2o::SE3Quat m = v2->estimate() * v1->estimate().inverse();

        g2o::EdgeSE3Expmap * e = new g2o::EdgeSE3Expmap();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setMeasurement(m);
        e->information() = Eigen::MatrixXd::Identity(6,6) * 100000000;
        data->optimizer.addEdge(e);
      }

      // Get camera intrinsics
      double focal_length = frame.intrinsics["focalLength"];
      Vector2d principle_point(frame.intrinsics["principlePoint"]["x"], frame.intrinsics["principlePoint"]["y"]);
      auto * cam_params = new g2o::CameraParameters(focal_length, principle_point, 0.);
      cam_params->setId(i+1);
      if (!data->optimizer.addParameter(cam_params)) {
        assert(false);
      }

      for (size_t j = 0; j < frame.landmarks.size(); j++) {
        
        size_t landmark_idx = frame.landmarks[j];
        Landmark &landmark = data->map.landmarkDatabase.landmarks[landmark_idx];
        if (landmark.sightings >= 3) {
          cv::KeyPoint keypoint = frame.kpts[j];
          Vector2d kp = Vector2d(keypoint.pt.x, keypoint.pt.y);

          g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(data->optimizer.vertex(landmark_idx)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(data->optimizer.vertex(frame_id)));
          e->setMeasurement(kp);
          e->information() = Matrix2d::Identity();
          e->setParameterId(0, i+1);
          // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
          // rk->setDelta(100.0);
          // e->setRobustKernel(rk);
          data->optimizer.addEdge(e);
        }
      }

      frame_id++;
    }
    
    printStats();
  }

  // Private methods

  void GraphConstruction::printStats() {
    int count = 0;
    for (size_t i = 0; i < data->map.landmarkDatabase.landmarks.size(); i++) {
      if (data->map.landmarkDatabase.landmarks[i].sightings >= 3) {
        count++;
      }
    }
    cout << "usable landmarks: " << count << endl;

    for (Frame const& frame : data->frames) {
      Projection projection(frame.frame_data);
      int count = 0;
      for (size_t i = 0; i < frame.landmarks.size(); i++) {
        size_t landmark_idx = frame.landmarks[i];
        Landmark &landmark = data->map.landmarkDatabase.landmarks[landmark_idx];
        if (landmark.sightings >= 3) {
          count++;
        }

        cv::Point2f pt = projection.projectToImage(landmark.position);
        cv::Point2f diff = frame.kpts[i].pt - pt;
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

  vector<size_t> GraphConstruction::getLandmarks(vector<cv::KeyPoint> &kpts, cv::Mat &desc, vector<float> &depth, json& frame_data) {
    // Filter out features that have been matched
    std::map<size_t, size_t> matches = getMatches(desc);
    Projection projection(frame_data);

    vector<size_t> landmarks;
    landmarks.reserve(kpts.size());

    for (size_t i = 0; i < kpts.size(); i++) {
      if (matches.find(i) == matches.end()) {
        // No match so create landmark
        Vector3d pt3d = projection.projectToWorld(kpts[i].pt, depth[i]);
        Landmark landmark(pt3d, kpts[i], desc.row(i));

        landmarks.push_back(data->map.landmarkDatabase.landmarks.size());
        data->map.landmarkDatabase.landmarks.push_back(landmark);
      } else {
        // We have a match so just push the match index
        landmarks.push_back(matches[i]);
        data->map.landmarkDatabase.landmarks[matches[i]].sightings++;
      }
    }

    return landmarks;
  }

  vector<float> GraphConstruction::getDepthValues(vector<cv::KeyPoint> &kpts, std::string depth_filepath, cv::Size img_size) {
    // Load depth map
    cout << "loading: " << depth_filepath << endl;
    cv::Mat depth = cv::imread(depth_filepath, cv::IMREAD_UNCHANGED);
    cv::resize(depth, depth, img_size, 0, 0, cv::INTER_LINEAR);

    vector<float> depth_values;
    depth_values.reserve(kpts.size());

    for (cv::KeyPoint const& kpt : kpts) {
      // Get depth value
      int idx_x = round(kpt.pt.x), idx_y = round(kpt.pt.y);
      float depth_value = depth.at<float>(idx_y, idx_x);
      depth_values.push_back(depth_value);
    }

    return depth_values;
  }

  std::map<size_t, size_t> GraphConstruction::getMatches(cv::Mat &desc) {
    // Get matches
    vector<cv::DMatch> matches = vision.match(desc, data->desc);
    cout << "matches: " << matches.size() << endl;

    // Populate `idx_matched` map
    std::map<size_t, size_t> idx_matched;
    for (size_t i = 0; i < matches.size(); i++) {
      int idx = matches[i].queryIdx;
      idx_matched[idx] = matches[i].trainIdx;
    }

    // Populate unmatched descriptions
    cv::Mat unmatched_desc; // TODO: see if there's a way to reserve capacity
    for (size_t i = 0; i < (unsigned)desc.rows; i++) {
      if (idx_matched.find(i) == idx_matched.end()) {
        unmatched_desc.push_back(desc.row(i));
      }
    }

    // Add new descriptions to `data->desc`
    if (data->desc.rows > 0) {
      cv::vconcat(data->desc, unmatched_desc, data->desc);
    } else {
      data->desc = unmatched_desc;
    }

    return idx_matched;
  }

  std::string GraphConstruction::getPathPrefix(int id, std::string directory) {
    std::string id_string = to_string(id);
    int zero_count = 8 - id_string.length();
    std::string prefix = std::string(zero_count, '0') + id_string + '_';
    return directory + '/' + prefix;
  }

  float GraphConstruction::angleDifference(float alpha, float beta) {
    float phi = fmod(abs(beta - alpha), 360.f);
    return phi > 180.f ? 360.f - phi : phi;
  }
}