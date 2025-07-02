// Only compile COLMAP refiner on desktop platforms (no SQLite on iOS/iPadOS)
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <sqlite3.h>

#include <opencv2/opencv.hpp>

#include "lar/processing/colmap_refiner.h"
#include "lar/core/utils/json.h"
#include "lar/mapping/location_matcher.h"

namespace lar {
  // map contains landmarks positioned by colmap, we will register images against the 3D point cloud
  // then use those localizations as initial pose for bundle adjustment
  // data->frames contains original ARKit data, we want to use the relative transforms between consecutive poses as odometry measurements
  ColmapRefiner::ColmapRefiner(std::shared_ptr<Mapper::Data> data) :
    data(data), tracker(data->map), bundle_adjustment(data), global_alignment(data) {
  }

  void ColmapRefiner::process() {
    // Update GPS alignment
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;
    localizations.clear();
    localizations.reserve(data->frames.size());

    // Localize frames
    for (auto& frame : data->frames) {
      std::cout << std::endl << "LOCALIZING FRAME " << frame.id << std::endl;
      std::string path_prefix = data->getPathPrefix(frame.id).string();
      std::string img_filepath = path_prefix + "image.jpeg";
      cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
      Eigen::Matrix4d extrinsics;
      if (tracker.localize(image, frame, extrinsics)) {
        std::cout << "extrinsics:" << extrinsics << std::endl;
        localizations.push_back(extrinsics);
      } else {
        localizations.push_back(frame.extrinsics);
      }
    }
    
  }

  void ColmapRefiner::processWithColmapData(const std::string& colmap_dir) {
    // Update GPS alignment
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;
    
    // Read COLMAP database and sparse reconstruction
    std::string database_path = colmap_dir + "/database.db";
    if (!colmap_db.readDatabase(database_path)) {
      std::cout << "Failed to read COLMAP database" << std::endl;
      return;
    }

    // Read from sparse reconstruction directory (aligned in place)
    std::string sparse_reconstruction_dir = colmap_dir + "/poses_txt";

    if (!colmap_db.readSparseReconstruction(sparse_reconstruction_dir)) {
      std::cout << "Failed to read COLMAP sparse reconstruction from " << sparse_reconstruction_dir << std::endl;
      return;
    }
    
    std::cout << "Loaded sparse reconstruction from " << sparse_reconstruction_dir << std::endl;

    // Construct complete landmarks from COLMAP data (like Python script)
    std::vector<Landmark> landmarks;
    colmap_db.constructLandmarksFromColmap(data->frames, landmarks, database_path);
    data->map.landmarks.insert(landmarks);

    // Extract camera poses for frames
    localizations.clear();
    localizations.reserve(data->frames.size());

    for (const auto& frame : data->frames) {
      std::string path_prefix = data->getPathPrefix(frame.id).string();
      std::string img_filename = std::filesystem::path(path_prefix + "image.jpeg").filename().string();
      
      const ColmapImage& colmap_img = colmap_db.images[frame.id+1];
      if (colmap_img.name == img_filename && colmap_img.positioned) {
        localizations.push_back(colmap_img.pose);
      } else {
        localizations.push_back(frame.extrinsics);
        std::cout << "No COLMAP pose found for frame " << frame.id << ", using ARKit pose" << std::endl;
      }
    }
  }

  void ColmapRefiner::optimize() {
    // Optimize
    bundle_adjustment.reset();
    // Construct first so that original ARKit transforms/extrinsics are used to compute odometry edges
    bundle_adjustment.construct(); 
    // Then we update the pose with our localizations before optimizing
    for (size_t i = 0; i < localizations.size(); i ++) {
      const Frame &frame = data->frames[i];
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
      g2o::SE3Quat pose = BundleAdjustment::poseFromExtrinsics(localizations[i]);
      v->setEstimate(pose);
    }
    bundle_adjustment.optimize();
  }

  void ColmapRefiner::rescale(double scale_factor) {
    if (scale_factor <= 0.0) {
      std::cout << "Invalid scale factor: " << scale_factor << std::endl;
      return;
    }
    
    std::cout << "Manual rescaling by factor: " << scale_factor << std::endl;
    
    // Ensure bundle adjustment is constructed before rescaling
    bundle_adjustment.reset();
    bundle_adjustment.construct();
    
    // Perform rescaling using the core implementation
    bundle_adjustment.performRescaling(scale_factor);
    
    // Update the data structures with scaled values
    bundle_adjustment.update();
    
    std::cout << "Manual rescaling complete" << std::endl;
  }

  void ColmapRefiner::saveMap(std::string dir) {
    std::filesystem::create_directory(dir);
    std::string output = dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    data->map.landmarks.cull();

    for (size_t i = 0; i < localizations.size(); i ++) {
      Frame &frame = data->frames[i];
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
      frame.extrinsics = BundleAdjustment::extrinsicsFromPose(v->estimate());
    }
    
    // Re-interpolate GPS observations using updated frame positions
    LocationMatcher temp_matcher;
    temp_matcher.matches = data->gps_obs;
    temp_matcher.reinterpolateMatches(data->frames);
    data->gps_obs = temp_matcher.matches;
    
    global_alignment.updateAlignment();

    // Serialize
    nlohmann::json frames_json = data->frames;
    std::ofstream(dir + "/frames.json") << frames_json << std::endl;
    
    nlohmann::json gps_json = data->gps_obs;
    std::ofstream(dir + "/gps.json") << gps_json << std::endl;

    nlohmann::json map_json = data->map;
    std::ofstream(dir + "/map.json") << map_json << std::endl;
  }

}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE
