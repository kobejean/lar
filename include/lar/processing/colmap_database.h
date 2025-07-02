#ifndef LAR_PROCESSING_COLMAP_DATABASE_H
#define LAR_PROCESSING_COLMAP_DATABASE_H

// Only compile COLMAP database support on desktop platforms
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <string>
#include <map>
#include <vector>
#include <sqlite3.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "lar/core/landmark.h"
#include "lar/core/landmark_database.h"
#include "lar/mapping/frame.h"

namespace lar {

  struct ColmapImage {
    int image_id;
    bool positioned;
    std::string name;
    Eigen::Matrix4d pose;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
  };

  struct ColmapPoint3D {
    int point3d_id;
    Eigen::Vector3d position;
    std::vector<std::pair<int, int>> track; // (image_id, point2d_idx) pairs
  };

  class ColmapDatabase {
  public:
    std::vector<ColmapImage> images;  // Indexed by image_id (0 is unused, IDs start from 1)
    std::vector<ColmapPoint3D> points3d;  // Indexed by point3d_id (0 is unused, IDs start from 1)

    bool readDatabase(const std::string& database_path);
    bool readSparseReconstruction(const std::string& sparse_dir);
    
    // Construct complete landmarks from COLMAP data (like Python script)
    void constructLandmarksFromColmap(
      const std::vector<Frame>& frames,
      std::vector<Landmark>& landmarks,
      const std::string& database_path
    );

  private:
    Eigen::Matrix4d colmapPoseToMatrix(const std::vector<double>& quat_trans);
    bool readImages(sqlite3* db);
    bool readKeypoints(sqlite3* db);
    bool readDescriptors(sqlite3* db);
    bool readPoints3D(const std::string& points3d_file);
    bool readImagesFromTxt(const std::string& images_file);
    
    // Helper functions for landmark construction
    cv::Mat getDescriptorForPoint(int point3d_id, const std::string& database_path);
    cv::Mat getDescriptorForPointCached(const ColmapPoint3D& point3d);
    Rect calculateSpatialBounds(const Eigen::Vector3d& position, const std::vector<Eigen::Vector3d>& camera_positions, double max_distance_factor = 1.5);
  };

}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE

#endif /* LAR_PROCESSING_COLMAP_DATABASE_H */