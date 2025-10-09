#!/usr/bin/env python3
"""
Streamlined script to process stereo vision data:
1. Copy *_image.jpeg files to a working directory
2. Extract SIFT features using either OpenCV or COLMAP
3. Run COLMAP to estimate camera positions using imported features
4. Integrate ARKit data for accurate intrinsics and metric scale
5. Export final scaled map.json with real SIFT descriptors
"""

import os
import shutil
import subprocess
import argparse
import glob
from pathlib import Path
from feature_extraction import extract_colmap_sift_features, extract_opencv_sift_features
from database_operations import create_colmap_database, export_poses, insert_two_view_geometries_from_arkit
from arkit_integration import load_arkit_data, create_reference_file_from_arkit
from map_export import export_aligned_map_json

def run_colmap_feature_matching(database_path):
    """Run COLMAP feature matching"""
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.num_threads", "8",
        # "--TwoViewGeometry.max_error", "6",
        # "--TwoViewGeometry.min_num_inliers", "12",
    ]
    
    print("Running COLMAP feature matching...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Feature matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Feature matching failed: {e}")
        return False

def run_colmap_vocab_tree_feature_matching(database_path, vocab_tree_path):
    """Run COLMAP vocab tree feature matching"""
    cmd = [
        "colmap", "vocab_tree_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.num_threads", "8",
        "--VocabTreeMatching.num_nearest_neighbors", "10",
        "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
    ]
    
    print("Running COLMAP vocab tree feature matching...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Feature matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Feature matching failed: {e}")
        return False

def run_colmap_mapping(database_path, output_dir):
    """Run COLMAP sparse reconstruction (mapping)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(Path(database_path).parent),
        "--output_path", str(output_path),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.extract_colors", "0",
        "--Mapper.num_threads", "8",
        "--Mapper.multiple_models", "0", # default: 1
        # "--Mapper.init_min_num_inliers", "50", # default: 100
        # "--Mapper.init_max_error", "6.0", # default: 4.0
        # "--Mapper.filter_max_reproj_error", "6.0", # default: 4.0
        # "--Mapper.tri_merge_max_reproj_error", "6.0", # default: 4.0
        # "--Mapper.tri_complete_max_reproj_error", "6.0", # default: 4.0
    ]

    print("Running COLMAP sparse reconstruction...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Sparse reconstruction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Sparse reconstruction failed: {e}")
        return False

def run_glomap_mapping(database_path, output_dir):
    """Run GLOMAP global structure-from-motion (faster alternative to COLMAP)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "glomap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(Path(database_path).parent),
        "--output_path", str(output_path),
        "--BundleAdjustment.optimize_intrinsics", "0",  # Don't refine intrinsics (we have ARKit calibration)
        "--skip_view_graph_calibration", "1",  # Skip calibration step (we have known intrinsics)
        "--skip_pruning", "1",  # Keep all points without pruning
    ]

    print("Running GLOMAP global reconstruction...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("GLOMAP reconstruction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"GLOMAP reconstruction failed: {e}")
        return False

def run_colmap_model_aligner(model_path, ref_file_path, database_path, max_error=0.1):
    """Run COLMAP model_aligner to geo-register the model using ARKit coordinates"""
    
    cmd = [
        "colmap", "model_aligner",
        "--input_path", str(model_path),
        "--output_path", str(model_path),  # Output to same directory to update in place
        # "--database_path", str(database_path),  # This will update the database with aligned poses
        "--ref_images_path", str(ref_file_path),
        "--ref_is_gps", "0",  # ARKit coordinates are cartesian, not GPS
        "--alignment_type", "custom",  # Use custom coordinate system (default but explicit)
        "--alignment_max_error", str(max_error),
        "--min_common_images", "3"  # Minimum images needed for alignment
    ]
    
    print("Running COLMAP model aligner for geo-registration...")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("Model alignment completed successfully - model updated in place")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Model alignment failed: {e}")
        print(f"Error output: {e.stderr}")
        if e.stdout:
            print(f"Standard output: {e.stdout}")
        return False

def run_colmap_feature_import(work_dir, database_path):
    """Import features from text files into COLMAP database"""
    cmd = [
        "colmap", "feature_importer",
        "--database_path", str(database_path),
        "--image_path", str(work_dir),
        "--import_path", str(work_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "0"
    ]
    
    print("Running COLMAP feature import...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Feature import completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Feature import failed: {e}")
        return False

def copy_images(source_dir, work_dir):
    """Copy all *_image.jpeg files from source to working directory"""
    source_path = Path(source_dir)
    work_path = Path(work_dir)
    
    # Create working directory if it doesn't exist
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files matching the pattern
    image_pattern = source_path / "*_image.jpeg"
    image_files = glob.glob(str(image_pattern))
    
    if not image_files:
        print(f"No *_image.jpeg files found in {source_dir}")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Copy images to working directory
    for img_file in image_files:
        filename = os.path.basename(img_file)
        dest_path = work_path / filename
        shutil.copy2(img_file, dest_path)
        print(f"Copied: {filename}")
    
    return True

def setup(args, frames_json_path, database_path, work_dir):
    arkit_frames = load_arkit_data(frames_json_path)
    create_colmap_database(arkit_frames, database_path, work_dir)
    if not copy_images(args.source_dir, work_dir):
        print("Failed to copy images. Exiting.")
        exit()
    return arkit_frames

def extract_features(args, work_dir, database_path):
    if args.use_colmap_sift:
        print(f"\nExtracting SIFT features using COLMAP (max {args.max_num_features} per image)...")
        
        # Run COLMAP feature extractor
        if not extract_colmap_sift_features(work_dir, database_path, args.max_num_features):
            print("Failed to extract SIFT features using COLMAP")
            exit(1)
    else:
        print(f"\nExtracting SIFT features using OpenCV (max {args.max_num_features} per image)...")
        if not extract_opencv_sift_features(work_dir, output_features=True, max_num_features=args.max_num_features):
            print("Failed to extract SIFT features using OpenCV")
            exit(1)
    
        # Import features into COLMAP database
        print("\nImporting features into COLMAP database...")
        if not run_colmap_feature_import(work_dir, database_path):
            print("COLMAP pipeline failed at feature import")
            exit(1)

def feature_matching(args, database_path):
    if args.use_vocab_tree:
        print("\nRunning vocab tree feature matching...")
        if not run_colmap_vocab_tree_feature_matching(database_path, Path(args.source_dir) / "vocab_tree.bin"):
            print("COLMAP pipeline failed at vocab tree feature matching")
            exit(1)
    else:
        print("\nRunning exhaustive feature matching...")
        if not run_colmap_feature_matching(database_path):
            print("COLMAP pipeline failed at feature matching")
            exit(1)

def insert_arkit_odometry(arkit_frames, database_path):
    """Insert ARKit relative poses as odometry constraints for bundle adjustment"""
    print("\nInserting ARKit VIO odometry constraints...")
    count = insert_two_view_geometries_from_arkit(arkit_frames, database_path)
    print(f"Added {count} relative pose constraints from ARKit VIO")

def sparse_reconstruction(args, database_path, sparse_dir):
    if args.use_glomap:
        print("\nRunning global reconstruction with GLOMAP...")
        if not run_glomap_mapping(database_path, sparse_dir):
            print("GLOMAP pipeline failed at reconstruction")
            exit(1)
    else:
        print("\nRunning sparse reconstruction with COLMAP...")
        if not run_colmap_mapping(database_path, sparse_dir):
            print("COLMAP pipeline failed at sparse reconstruction")
            exit(1)

    # Find reconstruction directory
    reconstruction_dirs = list(sparse_dir.glob("*"))
    if not reconstruction_dirs:
        print("No reconstruction found")
        exit(1)

    reconstruction_path = reconstruction_dirs[0]
    print(f"Found reconstruction in: {reconstruction_path}")
    return reconstruction_path

def model_alignment(args, arkit_frames, ref_coords_file, reconstruction_path, database_path):
    print("\nPerforming model alignment with ARKit coordinates...")
    
    # Create reference coordinates file
    if not create_reference_file_from_arkit(arkit_frames, ref_coords_file):
        print("Failed to create reference coordinates file")
        exit(1)
    
    # Run model aligner with database path to update poses in database and model in place
    if not run_colmap_model_aligner(reconstruction_path, ref_coords_file, database_path, args.alignment_max_error):
        print("Model alignment failed, proceeding with unaligned model")
    else:
        print("Model alignment completed successfully - database and model updated with aligned poses")
    
    return reconstruction_path

def export_map(args, database_path, map_json_file, arkit_frames, reconstruction_path, poses_dir):
    # Export poses from the final model
    if not export_poses(reconstruction_path, poses_dir):
        print("Failed to export poses")
        exit(1)
    
    # Export final map
    if not export_aligned_map_json(poses_dir, database_path, map_json_file,
                                   arkit_frames, Path(args.source_dir) / "map.json"):
        print("Failed to export aligned map.json")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Process stereo vision data with COLMAP/GLOMAP and ARKit integration")
    parser.add_argument("source_dir", help="Source directory containing *_image.jpeg files and frames.json")
    parser.add_argument("--use_colmap_sift", action="store_true",
                       help="Use COLMAP's built-in SIFT extractor instead of OpenCV")
    parser.add_argument("--max_num_features", type=int, default=16384,
                       help="Maximum number of features to extract per image (default: 16384)")
    parser.add_argument("--alignment_max_error", type=float, default=0.1,
                       help="Maximum error threshold for model alignment (default: 0.1)")
    parser.add_argument("--use_vocab_tree", action="store_true",
                       help="Use vocabulary tree matching instead of exhaustive matching")
    parser.add_argument("--use_glomap", action="store_true",
                       help="Use GLOMAP for reconstruction instead of COLMAP (faster, global SfM)")

    args = parser.parse_args()
    work_dir = Path(args.source_dir) / "colmap"
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    poses_dir = work_dir / "poses_txt"
    ref_coords_file = work_dir / "reference_coords.txt"
    map_json_file = work_dir / "map.json"
    frames_json_path = Path(args.source_dir) / "frames.json"

    # Step 1: Setup
    arkit_frames = setup(args, frames_json_path, database_path, work_dir)

    # Step 2: Extract SIFT features
    extract_features(args, work_dir, database_path)

    # Step 3: Feature matching
    feature_matching(args, database_path)

    # Step 4: Insert ARKit odometry constraints
    insert_arkit_odometry(arkit_frames, database_path)

    # Step 5: Sparse reconstruction
    reconstruction_path = sparse_reconstruction(args, database_path, sparse_dir)
    reconstruction_path = sparse_dir / "0"

    # Step 6: Model alignment
    reconstruction_path = model_alignment(args, arkit_frames, ref_coords_file, reconstruction_path, database_path)

    # Step 7: Export map
    export_map(args, database_path, map_json_file, arkit_frames, reconstruction_path, poses_dir)

    print(f"\n✅ Processing completed successfully!")
    print(f"Feature extraction method: {'COLMAP' if args.use_colmap_sift else 'OpenCV'}")
    print(f"Matching method: {'Vocabulary Tree' if args.use_vocab_tree else 'Exhaustive'}")
    print(f"Reconstruction method: {'GLOMAP' if args.use_glomap else 'COLMAP'}")
    print(f"Model alignment: Applied in place")
    print(f"Final map: {map_json_file}")
    print(f"Launch gui with: colmap gui --database_path {database_path} --import_path {reconstruction_path} --image_path {work_dir}")
    return 0

if __name__ == "__main__":
    exit(main())