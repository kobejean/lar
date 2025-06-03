#!/usr/bin/env python3
"""
Streamlined script to process stereo vision data:
1. Copy *_image.jpeg files to a working directory
2. Extract SIFT features using either OpenCV or COLMAP and write COLMAP-compatible text files
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
from database_operations import create_colmap_database, export_poses
from arkit_integration import load_arkit_data, create_reference_file_from_arkit
from map_export import export_aligned_map_json

def run_colmap_spatial_matching(database_path):
    """Run COLMAP spatial matching with optional ARKit pose priors"""
    cmd = [
        "colmap", "spatial_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1"
    ]
    
    # Use pose priors for spatial matching
    cmd.extend([
        "--SpatialMatching.ignore_z", "0",  # Don't ignore Z coordinate
        "--SpatialMatching.max_num_neighbors", "50",  # Increase neighbors for better coverage
        "--SpatialMatching.max_distance", "100.0"  # Max distance in ARKit units (meters)
    ])
    
    print(f"Running COLMAP spatial matching...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Spatial matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Spatial matching failed: {e}")
        return False

def run_colmap_feature_matching(database_path):
    """Run COLMAP feature matching"""
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1"
    ]
    
    print("Running COLMAP feature matching...")
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
        "--output_path", str(output_path)
    ]
    
    print("Running COLMAP sparse reconstruction...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Sparse reconstruction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Sparse reconstruction failed: {e}")
        return False

def run_colmap_model_aligner(input_model_path, output_model_path, ref_file_path, max_error=0.1):
    """Run COLMAP model_aligner to geo-register the model using ARKit coordinates"""
    output_model_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "colmap", "model_aligner",
        "--input_path", str(input_model_path),
        "--output_path", str(output_model_path),
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
        print("Model alignment completed successfully")
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

def main():
    parser = argparse.ArgumentParser(description="Process stereo vision data with COLMAP and ARKit integration")
    parser.add_argument("source_dir", help="Source directory containing *_image.jpeg files and frames.json")
    parser.add_argument("--use_colmap_sift", action="store_true",
                       help="Use COLMAP's built-in SIFT extractor instead of OpenCV")
    parser.add_argument("--use_spatial_matching", action="store_true",
                       help="Use spatial matching instead of exhaustive matching")
    parser.add_argument("--pose_prior_variance", type=float, default=25.0,
                       help="Pose prior variance for spatial matching (default: 25.0)")
    parser.add_argument("--max_num_features", type=int, default=8192,
                       help="Maximum number of features to extract per image (default: 8192)")
    parser.add_argument("--alignment_max_error", type=float, default=0.1,
                       help="Maximum error threshold for model alignment (default: 0.1)")
    parser.add_argument("--launch_gui", action="store_true",
                       help="Launch COLMAP GUI after reconstruction completes")
    
    args = parser.parse_args()
    
    work_dir = Path(args.source_dir) / "colmap"
    database_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    aligned_dir = work_dir / "aligned"
    poses_dir = work_dir / "poses_txt"
    ref_coords_file = work_dir / "reference_coords.txt"
    map_json_file = work_dir / "map_aligned.json"
        
    # Load ARKit data if available
    arkit_frames = None
    frames_json_path = Path(args.source_dir) / "frames.json"
    if frames_json_path.exists():
        arkit_frames = load_arkit_data(frames_json_path)
        
        # Create database with ARKit cameras and pose priors
        from database_operations import create_colmap_database_with_pose_priors
        if args.use_spatial_matching:
            print(f"Creating database with ARKit pose priors (variance: {args.pose_prior_variance})")
            create_colmap_database_with_pose_priors(
                arkit_frames, database_path, work_dir, 
                pose_variance=args.pose_prior_variance
            )
        elif not args.use_colmap_sift:
            # Regular database creation for exhaustive matching
            create_colmap_database(arkit_frames, database_path, work_dir)
    else:
        print(f"Warning: ARKit frames file not found at {frames_json_path}")
        print("Proceeding without ARKit alignment")
    
    # Step 1: Copy images
    if not copy_images(args.source_dir, work_dir):
        print("Failed to copy images. Exiting.")
        return 1
    
    # Step 2: Extract SIFT features to text files
    if args.use_colmap_sift:
        print(f"\nExtracting SIFT features using COLMAP (max {args.max_num_features} per image)...")
        
        # Run COLMAP feature extractor (saves to text files)
        if not extract_colmap_sift_features(work_dir, database_path, args.max_num_features):
            print("Failed to extract SIFT features using COLMAP")
            return 1
    else:
        print(f"\nExtracting SIFT features using OpenCV (max {args.max_num_features} per image)...")
        if not extract_opencv_sift_features(work_dir, output_features=True, max_num_features=args.max_num_features):
            print("Failed to extract SIFT features using OpenCV")
            return 1
    
    # Step 3: Import features into COLMAP database (unified for both methods)
    print("\nImporting features into COLMAP database...")
    if not run_colmap_feature_import(work_dir, database_path):
        print("COLMAP pipeline failed at feature import")
        return 1
    
    # Step 4: Feature matching (spatial vs exhaustive)
    if args.use_spatial_matching:
        print("\nRunning spatial feature matching...")
        if not run_colmap_spatial_matching(database_path):
            print("COLMAP pipeline failed at spatial matching")
            return 1
    else:
        print("\nRunning exhaustive feature matching...")
        if not run_colmap_feature_matching(database_path):
            print("COLMAP pipeline failed at feature matching")
            return 1
    
    # Step 5: Sparse reconstruction
    print("\nRunning sparse reconstruction...")
    if not run_colmap_mapping(database_path, sparse_dir):
        print("COLMAP pipeline failed at sparse reconstruction")
        return 1
    
    # Find reconstruction directory
    reconstruction_dirs = list(sparse_dir.glob("*"))
    if not reconstruction_dirs:
        print("No reconstruction found")
        return 1
    
    reconstruction_path = reconstruction_dirs[0]
    print(f"Found reconstruction in: {reconstruction_path}")
    
    # Step 6: Model alignment (if ARKit data available)
    aligned_reconstruction_path = reconstruction_path
    if arkit_frames:
        print("\nPerforming model alignment with ARKit coordinates...")
        
        # Create reference coordinates file
        if not create_reference_file_from_arkit(arkit_frames, ref_coords_file):
            print("Failed to create reference coordinates file")
            return 1
        
        # Run model aligner
        if not run_colmap_model_aligner(reconstruction_path, aligned_dir, ref_coords_file, args.alignment_max_error):
            print("Model alignment failed, proceeding with unaligned model")
            aligned_reconstruction_path = reconstruction_path
        else:
            aligned_reconstruction_path = aligned_dir
            print("Model alignment completed successfully")
    
    # Export poses from the final model (aligned or original)
    if not export_poses(aligned_reconstruction_path, poses_dir):
        print("Failed to export poses")
        return 1
    
    # Export final map
    if not export_aligned_map_json(poses_dir, work_dir, map_json_file):
        print("Failed to export aligned map.json")
        return 1
    
    print(f"\n✅ Processing completed successfully!")
    print(f"Feature extraction method: {'COLMAP' if args.use_colmap_sift else 'OpenCV'}")
    print(f"Matching method: {'Spatial' if args.use_spatial_matching else 'Exhaustive'}")
    print(f"Model alignment: {'Applied' if arkit_frames and aligned_reconstruction_path == aligned_dir else 'Not applied'}")
    print(f"Final map: {map_json_file}")
    
    # Launch GUI if requested
    if args.launch_gui:
        print(f"\nLaunching COLMAP GUI...")
        try:
            gui_cmd = [
                "colmap", "gui",
                "--database_path", str(database_path),
                "--import_path", str(aligned_reconstruction_path),
                "--image_path", str(work_dir)
            ]
            subprocess.Popen(gui_cmd)
            print("COLMAP GUI launched successfully")
        except Exception as e:
            print(f"Failed to launch GUI: {e}")
            print(f"You can manually launch with: colmap gui --database_path {database_path} --import_path {aligned_reconstruction_path} --image_path {work_dir}")
    return 0

if __name__ == "__main__":
    exit(main())