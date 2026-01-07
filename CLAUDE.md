# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building
- `make all` - Standard build (Release mode)
- `make fast` - Parallel build (Release mode, -j 8)
- `make debug` - Debug build
- `make compact` - Minimal build with LAR_COMPACT_BUILD=ON
- `make service` - Build with gRPC service (LAR_BUILD_SERVICE=ON)
- `make clean` - Clean build directory

### Testing
- `make tests` - Build with tests enabled
- `./bin/lar_test` - Run the test suite

### Frameworks & Artifacts
- `make frameworks` - Build iOS/macOS frameworks using build_frameworks.bash
- `make artifacts` - Build release artifacts (calls build_artifacts.bash)

### Applications
Built applications are in `bin/`:
- `lar_create_map` - Creates 3D maps from image datasets
- `lar_localize` - Localizes poses within existing maps
- `lar_refine_colmap` - COLMAP integration for map refinement
- `lar_rtree` - Spatial indexing utilities
- `lar_server` - gRPC navigation server

## Architecture Overview

LAR is structured around a 4-stage computer vision pipeline:

### Core Components
- **LandmarkDatabase** (`lar/core/landmark_database.h`) - Spatial indexing of 3D landmarks using RegionTree
- **Map** (`lar/core/map.h`) - Container for anchors and landmark database
- **Landmark** (`lar/core/landmark.h`) - 3D feature points with descriptors and observations
- **Anchor** (`lar/core/anchor.h`) - Reference coordinate frames for map alignment

### Processing Pipeline
1. **Mapping** (`lar/mapping/`) - Processes image sequences into maps
   - **Mapper** - Main orchestrator that processes frames and GPS data
   - **Frame** - Camera pose and metadata container
   - **LocationMatcher** - GPS/location integration
   
2. **Processing** (`lar/processing/`) - Optimization and refinement
   - **BundleAdjustment** - g2o-based optimization of poses and landmarks
   - **MapProcessor** - High-level processing coordinator  
   - **ColmapRefiner** - Integration with COLMAP for photogrammetry
   
3. **Tracking** (`lar/tracking/`) - Real-time localization
   - **Tracker** - Main localization engine using feature matching
   - **Vision** - Feature extraction and matching utilities

4. **Core Utilities** (`lar/core/`)
   - **RegionTree** - R-tree spatial indexing for fast landmark queries
   - **WGS84/GPS** utilities for coordinate conversion
   - JSON serialization support throughout

### Data Flow
1. Input images → Mapper processes frames → extracts features
2. Bundle adjustment optimizes pose graph using g2o
3. Map exported as JSON with landmarks in spatial index
4. Tracker loads map and localizes new images via feature matching

### Service Module (`lar/service/`)
gRPC service layer exposing LAR functionality over the network:
- **NavigationService** - Exposes `Map::getPath` A* pathfinding via gRPC
- Proto definitions in `proto/navigation.proto`
- Built with `LAR_BUILD_SERVICE=ON` (enabled by default)
- Supports gRPC reflection for debugging with `grpcurl`

## Key Dependencies
- **OpenCV 4.5.4+** - Feature extraction, image processing
- **Eigen3 3.4.90+** - Linear algebra and transformations
- **g2o 1.0.0+** - Graph optimization for bundle adjustment
- **nlohmann_json 3.11.3+** - JSON serialization
- **gRPC** - Remote procedure calls for service module
- **protobuf** - Protocol buffer serialization for gRPC
- **COLMAP** (optional) - External photogrammetry pipeline integration

## COLMAP Integration
The `script/colmap/` directory contains Python utilities for COLMAP integration:
- Feature extraction using SIFT
- Database operations for COLMAP workflows  
- ARKit integration for mobile data
- Map export with coordinate alignment

## Platform Support
- Primary development: macOS (ARM64/Apple Silicon)
- CI testing: Ubuntu Latest, macOS Latest
- Builds iOS/macOS XCFrameworks for Swift integration
- Uses Apple Clang on macOS with explicit compiler flags

## Testing
- Tests built with `LAR_BUILD_TESTS=ON`
- CI runs `make tests && ./bin/lar_test`
- Test framework in `test/all_tests.cpp`

## Thread Safety

### Current Architecture
- **LandmarkDatabase**: Thread-safe using `std::shared_mutex` (readers-writer lock)
  - All public methods are protected with appropriate locks
  - Safe for concurrent access from multiple threads
- **RegionTree**: NOT thread-safe (by design, lock-free)
  - Must be accessed through `LandmarkDatabase` for concurrent usage
  - Direct usage requires external synchronization
  - Lock-free design prevents deadlocks when called from `LandmarkDatabase`

### Usage in AR Applications
Safe patterns for background localization:
- ✅ Multiple threads reading landmarks simultaneously
- ✅ Concurrent `addObservation()` calls from different threads
- ✅ One thread writing while others read

Pointer stability: Pointers from `insert()` remain valid across `updateBounds()` and tree rebalancing (tested in `test/core/spatial/region_tree_test.cpp`).

## Comment From Lead Developer
The project is in an experimental phase and not structured ideally. We are trying different ways of mapping and localization and code is frequently being changed. Once we have a good idea of how to approach mapping and localization, we will refactor and optimize the code. 

Data is collected on an iPad Pro with a custom app that uses ARKit. It generates files like frames.json, map.json, gps.json as well as images, depth maps and depth map confidence values.

Since we are mapping an area the size of a park, depth maps turned out to not be usefull as we can't get good enough metric depth values, so this data is largely being ignored. Doing mapping from scrach turned out to bee too difficult to do right away. Our current strategy is to rely heavily on COLMAP, ARKit, OpenCV, g2o etc. and then gradually customize the pipeline with our own implementations.

Currently use the colmap python script to use ARKit data to create a rough map with camera poses and landmarks. Then to incorporate ARKit's high quqlity ralative transforms, I use the relative poses as odometry measurements for a final global bundle adjustment.