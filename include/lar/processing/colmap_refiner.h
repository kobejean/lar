#ifndef LAR_PROCESSING_COLMAP_REFINER_H
#define LAR_PROCESSING_COLMAP_REFINER_H

// Only compile COLMAP refiner on desktop platforms (no SQLite on iOS/iPadOS)
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <string>

#include <Eigen/Core>

#include "lar/core/map.h"
#include "lar/mapping/mapper.h"
#include "lar/processing/bundle_adjustment.h"
#include "lar/processing/frame_processor.h"
#include "lar/processing/global_alignment.h"
#include "lar/processing/colmap_database.h"
#include "lar/tracking/tracker.h"

namespace lar {

  class ColmapRefiner {
    public:
      std::shared_ptr<Mapper::Data> data;
      
      ColmapRefiner(std::shared_ptr<Mapper::Data> data);
      void process();
      void processWithColmapData(const std::string& colmap_dir);
      void optimize();
      void rescale(double scale_factor);
      void saveMap(std::string dir);
    // private:
      std::vector<Eigen::Matrix4d> localizations;
      Tracker tracker;
      BundleAdjustment bundle_adjustment;
      GlobalAlignment global_alignment;
      
      // COLMAP data processing
      ColmapDatabase colmap_db;
      // FrameProcessor frame_processor;
  };
}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE

#endif /* LAR_PROCESSING_COLMAP_REFINER_H */