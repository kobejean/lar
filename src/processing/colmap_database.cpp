// Only compile COLMAP database support on desktop platforms
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <Eigen/Geometry>

#include "lar/processing/colmap_database.h"

namespace lar {

  bool ColmapDatabase::readDatabase(const std::string& database_path) {
    sqlite3* db;
    int rc = sqlite3_open(database_path.c_str(), &db);
    
    if (rc) {
      std::cout << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
      return false;
    }


    bool success = readImages(db) && readKeypoints(db) && readDescriptors(db);
    
    sqlite3_close(db);
    return success;
  }

  bool ColmapDatabase::readSparseReconstruction(const std::string& sparse_dir) {
    std::string images_txt = sparse_dir + "/images.txt";
    std::string points3d_txt = sparse_dir + "/points3D.txt";
    
    if (!std::filesystem::exists(images_txt) || !std::filesystem::exists(points3d_txt)) {
      std::cout << "Text files found in " << sparse_dir << std::endl;
      return false;
    }
    
    std::cout << "Reading from text files..." << std::endl;
    return readImagesFromTxt(images_txt) && readPoints3D(points3d_txt);
  }

  void ColmapDatabase::constructLandmarksFromColmap(
    const std::vector<Frame>& frames,
    std::vector<Landmark>& landmarks,
    const std::string& database_path
  ) {
    std::cout << "Constructing landmarks from COLMAP data..." << std::endl;
    
    // Pre-cache all descriptors to avoid repeated SQLite queries
    std::cout << "Pre-loading all descriptors..." << std::endl;
    sqlite3* db;
    int rc = sqlite3_open(database_path.c_str(), &db);
    if (rc == SQLITE_OK) {
      // All descriptors are already loaded into images vector during readDescriptors()
      sqlite3_close(db);
    }
    
    int landmarks_created = 0;
    int observations_created = 0;

    // For each 3D point, construct a complete landmark
    for (const auto& point3d : points3d) {
      int point3d_id = point3d.point3d_id;
      
      // Convert COLMAP coordinates to ARKit convention (y and z flipped)
      Eigen::Vector3d position;
      position.x() = point3d.position.x();
      position.y() = -point3d.position.y();  // Flip Y
      position.z() = -point3d.position.z();  // Flip Z

      // Get descriptor for this 3D point from cached data
      cv::Mat descriptor = getDescriptorForPointCached(point3d);
      if (descriptor.empty()) {
        if (landmarks_created == 0) { // Only print for first few
          std::cout << "No descriptor found for point3d_id " << point3d_id << std::endl;
        }
        continue; // Skip if no descriptor found
      }

      // Collect camera positions that observe this point
      std::vector<Eigen::Vector3d> observing_camera_positions;
      std::vector<Landmark::Observation> observations;

      for (const auto& [image_id, point2d_idx] : point3d.track) {
        if (image_id >= static_cast<int>(images.size()) || image_id <= 0) {
          if (landmarks_created == 0) std::cout << "Image " << image_id << " not found in images vector" << std::endl;
          continue;
        }

        const ColmapImage& colmap_img = images[image_id];
        const Frame& frame = frames[image_id-1];

        // Get camera position
        Eigen::Vector3d camera_pos = frame.extrinsics.block<3,1>(0,3);
        observing_camera_positions.push_back(camera_pos);

        // Get the keypoint for this observation
        const cv::KeyPoint& kpt = colmap_img.keypoints[point2d_idx];
        
        // Calculate depth from camera position to landmark
        Eigen::Vector3d distance_vec = position - camera_pos;
        float depth = static_cast<float>(distance_vec.norm());

        // Create observation
        Landmark::Observation obs{
          .frame_id = static_cast<size_t>(frame.id),
          .timestamp = frame.timestamp,
          .cam_pose = frame.extrinsics,
          .kpt = kpt,
          .depth = depth,
          .depth_confidence = 0.0f, // COLMAP doesn't provide depth confidence
          .surface_normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f), // Default orientation
        };
        
        observations.push_back(obs);
        observations_created++;
        
        if (landmarks_created == 0) std::cout << "Created observation for frame " << frame.id << std::endl;
      }

      if (observations.empty()) {
        if (landmarks_created == 0) std::cout << "No observations created for point3d_id " << point3d_id << std::endl;
        continue;
      }

      // Create complete landmark
      Landmark landmark(position, descriptor, point3d_id);
      landmark.orientation = Eigen::Vector3f(0.0f, 0.0f, 1.0f); // Default orientation like Python script
      landmark.updateBounds(observing_camera_positions);
      landmark.sightings = observations.size();
      landmark.obs = observations;

      landmarks.push_back(landmark);
      landmarks_created++;
      if (landmarks_created % 10000 == 0) {
        std::cout << "landmarks_created: " << landmarks_created << std::endl;
      }
    }

    std::cout << "Created " << landmarks_created << " landmarks with " << observations_created << " observations" << std::endl;
  }

  // Private methods

  cv::Mat ColmapDatabase::getDescriptorForPointCached(const ColmapPoint3D& point3d) {
    // Get descriptor from the first image in track that has descriptors cached
    for (const auto& [img_id, point2d_idx] : point3d.track) {
      if (img_id < static_cast<int>(images.size()) && 
          !images[img_id].descriptors.empty() && 
          point2d_idx < images[img_id].descriptors.rows) {
        return images[img_id].descriptors.row(point2d_idx).clone();
      }
    }
    return cv::Mat(); // No descriptor found
  }

  cv::Mat ColmapDatabase::getDescriptorForPoint(int point3d_id, const std::string& database_path) {
    // Binary search for the 3D point with matching ID
    auto it = std::lower_bound(points3d.begin(), points3d.end(), point3d_id,
                               [](const ColmapPoint3D& point, int id) {
                                 return point.point3d_id < id;
                               });
    
    if (it == points3d.end() || it->point3d_id != point3d_id || it->track.empty()) {
      std::cout << "Point3D " << point3d_id << " not found or has empty track" << std::endl;
      return cv::Mat();
    }

    const auto& track = it->track;
    
    static bool first_call = true;
    if (first_call) {
      std::cout << "First descriptor extraction: point3d_id=" << point3d_id << ", track size=" << track.size() << std::endl;
      first_call = false;
    }

    // Try to get descriptor from first image that has descriptors
    sqlite3* db;
    int rc = sqlite3_open(database_path.c_str(), &db);
    if (rc) {
      std::cout << "Failed to open database for descriptor extraction" << std::endl;
      return cv::Mat();
    }

    for (const auto& [img_id, point2d_idx] : track) {
      if (first_call) {
        std::cout << "Trying to get descriptor for image_id=" << img_id << ", point2d_idx=" << point2d_idx << std::endl;
      }
      
      const char* sql = "SELECT rows, cols, data FROM descriptors WHERE image_id = ?";
      sqlite3_stmt* stmt;
      
      rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
      if (rc != SQLITE_OK) {
        if (first_call) std::cout << "Failed to prepare descriptor query" << std::endl;
        continue;
      }

      sqlite3_bind_int(stmt, 1, img_id);

      if (sqlite3_step(stmt) == SQLITE_ROW) {
        if (first_call) std::cout << "Found descriptors for image_id=" << img_id << std::endl;
        int desc_rows = sqlite3_column_int(stmt, 0);
        int desc_cols = sqlite3_column_int(stmt, 1);
        const void* data = sqlite3_column_blob(stmt, 2);
        int data_size = sqlite3_column_bytes(stmt, 2);

        if (first_call) {
          std::cout << "Descriptor data: rows=" << desc_rows << ", cols=" << desc_cols 
                    << ", data_size=" << data_size << ", point2d_idx=" << point2d_idx << std::endl;
        }
        
        if (data && point2d_idx < desc_rows) {
          // Extract descriptor for this specific keypoint
          const uint8_t* desc_data = static_cast<const uint8_t*>(data);
          cv::Mat descriptors(desc_rows, desc_cols, CV_8UC1);
          std::memcpy(descriptors.data, desc_data, data_size);
          
          cv::Mat descriptor = descriptors.row(point2d_idx).clone();
          sqlite3_finalize(stmt);
          sqlite3_close(db);
          if (first_call) std::cout << "Successfully extracted descriptor!" << std::endl;
          return descriptor;
        } else {
          if (first_call) std::cout << "Invalid point2d_idx or no data" << std::endl;
        }
      } else {
        if (first_call) std::cout << "No descriptors found for image_id=" << img_id << std::endl;
      }
      sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
    
    // TEMPORARY: Create dummy descriptor for testing
    std::cout << "No descriptor found for point3d_id " << point3d_id << ", creating dummy descriptor" << std::endl;
    cv::Mat dummy_descriptor = cv::Mat::zeros(1, 128, CV_8UC1);
    for (int i = 0; i < 128; i++) {
      dummy_descriptor.at<uint8_t>(0, i) = i % 256;
    }
    return dummy_descriptor;
  }


  Eigen::Matrix4d ColmapDatabase::colmapPoseToMatrix(const std::vector<double>& quat_trans) {
    double qw = quat_trans[0];
    double qx = quat_trans[1]; 
    double qy = quat_trans[2];
    double qz = quat_trans[3];
    double tx = quat_trans[4];
    double ty = quat_trans[5];
    double tz = quat_trans[6];

    // Convert quaternion to rotation matrix
    Eigen::Quaternion<double> quat(qw, qx, qy, qz);
    Eigen::Matrix3d R = quat.toRotationMatrix();
    
    // Calculate camera position: -R^T * t (COLMAP world-to-camera to camera position)
    Eigen::Vector3d colmap_translation(tx, ty, tz);
    Eigen::Vector3d camera_pos = -R.transpose() * colmap_translation;
    
    // Apply coordinate system conversion (COLMAP to ARKit)
    Eigen::Vector3d arkit_position;
    arkit_position.x() = camera_pos.x();
    arkit_position.y() = -camera_pos.y();  // Flip Y
    arkit_position.z() = -camera_pos.z();  // Flip Z
    
    // Build ARKit convention camera-to-world matrix
    // Following the Python implementation in camera_to_world_matrix property
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    
    // Set rotation part with coordinate system conversion
    transform(0,0) =  R(0,0); transform(0,1) =  -R(1,0); transform(0,2) =  -R(2,0);
    transform(1,0) =  -R(0,1); transform(1,1) =  R(1,1); transform(1,2) =  R(2,1);
    transform(2,0) =  -R(0,2); transform(2,1) =  R(1,2); transform(2,2) =  R(2,2);
    
    // Set translation part
    transform(0,3) = arkit_position.x();
    transform(1,3) = arkit_position.y();
    transform(2,3) = arkit_position.z();
    
    return transform;
  }

  bool ColmapDatabase::readImages(sqlite3* db) {
    // Read image names only - poses will come from sparse reconstruction
    const char* sql = "SELECT image_id, name FROM images";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
      std::cout << "SQL error preparing images query: " << sqlite3_errmsg(db) << std::endl;
      return false;
    }

    int count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      ColmapImage img;
      img.image_id = sqlite3_column_int(stmt, 0);
      img.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
      img.positioned = false;
      // Note: pose will be set later from sparse reconstruction
      
      // Resize vector if needed
      if (img.image_id >= static_cast<int>(images.size())) {
        images.resize(img.image_id + 1);
      }
      images[img.image_id] = img;
      count++;
    }

    sqlite3_finalize(stmt);
    std::cout << "Read " << count << " images from database (poses will come from sparse reconstruction)" << std::endl;
    return true;
  }

  bool ColmapDatabase::readKeypoints(sqlite3* db) {
    const char* sql = "SELECT image_id, data FROM keypoints";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
      std::cout << "SQL error preparing keypoints query: " << sqlite3_errmsg(db) << std::endl;
      return false;
    }

    int count = 0;
    int total_keypoints = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      int image_id = sqlite3_column_int(stmt, 0);
      const void* data = sqlite3_column_blob(stmt, 1);
      int data_size = sqlite3_column_bytes(stmt, 1);
      
      if (data_size > 0) {
        // Parse COLMAP keypoint format (6 floats per keypoint: x, y, scale, orientation, data, data)
        const float* kpt_data = static_cast<const float*>(data);
        int num_keypoints = data_size / (6 * sizeof(float));
        
        // Resize vector if needed
        if (image_id >= static_cast<int>(images.size())) {
          images.resize(image_id + 1);
        }
        images[image_id].image_id = image_id;
        
        for (int i = 0; i < num_keypoints; i++) {
          cv::KeyPoint kpt;
          kpt.pt.x = kpt_data[i * 6 + 0];
          kpt.pt.y = kpt_data[i * 6 + 1];
          kpt.size = kpt_data[i * 6 + 2];
          kpt.angle = kpt_data[i * 6 + 3];
          images[image_id].keypoints.push_back(kpt);
        }
        total_keypoints += num_keypoints;
        count++;
      }
    }

    sqlite3_finalize(stmt);
    std::cout << "Read keypoints for " << count << " images, total " << total_keypoints << " keypoints" << std::endl;
    
    return count > 0;
  }

  bool ColmapDatabase::readDescriptors(sqlite3* db) {
    const char* sql = "SELECT image_id, data FROM descriptors";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
      std::cout << "SQL error preparing descriptors query: " << sqlite3_errmsg(db) << std::endl;
      return false;
    }

    int count = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      int image_id = sqlite3_column_int(stmt, 0);
      const void* data = sqlite3_column_blob(stmt, 1);
      int data_size = sqlite3_column_bytes(stmt, 1);
      
      // Resize vector if needed
      if (image_id >= static_cast<int>(images.size())) {
        images.resize(image_id + 1);
      }
      if (data_size > 0) {
        // COLMAP descriptors are typically 128-dimensional SIFT descriptors (uint8)
        int num_descriptors = images[image_id].keypoints.size();
        int descriptor_size = 128; // SIFT descriptor size
        
        if (data_size == num_descriptors * descriptor_size) {
          cv::Mat descriptors(num_descriptors, descriptor_size, CV_8UC1);
          std::memcpy(descriptors.data, data, data_size);
          images[image_id].descriptors = descriptors;
          count++;
        }
      }
    }

    sqlite3_finalize(stmt);
    std::cout << "Read descriptors for " << count << " images" << std::endl;
    return true;
  }

  bool ColmapDatabase::readPoints3D(const std::string& points3d_file) {
    std::ifstream file(points3d_file);
    if (!file.is_open()) {
      std::cout << "Failed to open points3D file: " << points3d_file << std::endl;
      return false;
    }

    std::string line;
    int count = 0;
    
    // Skip header comments
    while (std::getline(file, line) && line[0] == '#') {}
    
    do {
      if (line.empty()) continue;
      
      std::istringstream iss(line);
      ColmapPoint3D point;
      double r, g, b; // color values we'll ignore
      double error;   // reconstruction error we'll ignore
      
      if (!(iss >> point.point3d_id >> point.position.x() >> point.position.y() >> point.position.z() >> r >> g >> b >> error)) {
        continue;
      }

      // Parse track (image_id, point2d_idx pairs)
      int image_id, point2d_idx;
      while (iss >> image_id >> point2d_idx) {
        point.track.emplace_back(image_id, point2d_idx);
      }

      points3d.push_back(point);
      count++;
      
    } while (std::getline(file, line));

    file.close();
    std::cout << "Read " << count << " 3D points from " << points3d_file << std::endl;
    return count > 0;
  }

  bool ColmapDatabase::readImagesFromTxt(const std::string& images_file) {
    std::ifstream file(images_file);
    if (!file.is_open()) {
      std::cout << "Failed to open images file: " << images_file << std::endl;
      return false;
    }

    std::string line;
    int count = 0;
    
    // Skip header comments
    while (std::getline(file, line) && line[0] == '#') {}
    
    do {
      if (line.empty()) continue;
      
      std::istringstream iss(line);
      int image_id;
      double qw, qx, qy, qz, tx, ty, tz;
      int camera_id;
      
      if (!(iss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id)) {
        continue;
      }
      
      // Read the rest of the line as the image name (may contain spaces)
      std::string name;
      std::getline(iss, name);
      // Trim leading whitespace
      size_t start = name.find_first_not_of(" \t");
      if (start != std::string::npos) {
        name = name.substr(start);
      }
      
      // Debug: print the first few image names to see the format
      if (count < 5) {
        std::cout << "Read image_id=" << image_id << ", name='" << name << "'" << std::endl;
      }

      std::vector<double> quat_trans = {qw, qx, qy, qz, tx, ty, tz};
      Eigen::Matrix4d pose = colmapPoseToMatrix(quat_trans);
      
      // Resize vector if needed
      if (image_id >= static_cast<int>(images.size())) {
        images.resize(image_id + 1);
      }
      // Update existing image or create new one
      images[image_id].pose = pose;
      images[image_id].name = name;
      images[image_id].image_id = image_id;
      images[image_id].positioned = true;
      
      count++;
      
      // Skip the next line (feature points line)
      std::getline(file, line);
      
    } while (std::getline(file, line));

    file.close();
    std::cout << "Read " << count << " camera poses from " << images_file << std::endl;
    return count > 0;
  }

}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE