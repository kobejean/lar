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
from database_operations import create_colmap_database, export_poses, create_arkit_seed_model
from arkit_integration import load_arkit_data, create_reference_file_from_arkit
from map_export import export_aligned_map_json

# Default vocab tree bundled alongside this script (Flickr100K, 32K words).
BUNDLED_VOCAB_TREE = Path(__file__).resolve().parent / "vocab_tree.bin"

def resolve_vocab_tree_path(source_dir):
    """Resolve which vocab tree to use.

    A per-session tree at <source_dir>/vocab_tree.bin takes precedence (lets you
    override with one trained on your own data); otherwise fall back to the
    bundled default committed next to this script. Returns None if neither exists.
    """
    session_tree = Path(source_dir) / "vocab_tree.bin"
    if session_tree.exists():
        return session_tree
    if BUNDLED_VOCAB_TREE.exists():
        return BUNDLED_VOCAB_TREE
    return None

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

def run_colmap_sequential_feature_matching(database_path, overlap, vocab_tree_path=None):
    """Run COLMAP sequential feature matching.

    Matches each image against the next `overlap` images in filename order
    (images are named `{frame_id:08d}_image.jpeg`, so lexicographic order is
    capture order). When a vocab tree is available, loop detection is enabled to
    additionally match images that are near in space but far apart in sequence.
    """
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.num_threads", "8",
        # Tight geometric verification so the extra loop-closure candidate edges
        # that survive are reliable, not spurious.
        "--SiftMatching.max_ratio", "0.8",
        "--TwoViewGeometry.min_num_inliers", "15",
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
    ]

    if vocab_tree_path is not None:
        cmd += [
            "--SequentialMatching.loop_detection", "1",
            # Thorough loop closure with soft visual-word assignment
            # (num_nearest_neighbors 5) for high retrieval recall. Query every 5th
            # image rather than every image: consecutive frames are near-identical
            # so they retrieve the same loop candidates, making period 1 mostly
            # redundant work (a revisit spans many frames, so period 5 still catches
            # it) -- period 5 is ~5x faster with negligible recall loss.
            "--SequentialMatching.loop_detection_period", "5",
            "--SequentialMatching.loop_detection_num_images", "40",
            "--SequentialMatching.loop_detection_num_nearest_neighbors", "5",
            # Cap features used to build/query the vocab tree index (default -1 =
            # all). Indexing every extracted feature (up to max_num_features) is
            # the dominant cost; top-scale features dominate retrieval anyway, so
            # this speeds indexing ~4-8x. Full pairwise matching still uses all
            # features, so map density is unaffected.
            "--SequentialMatching.loop_detection_max_num_features", "4096",
            "--SequentialMatching.vocab_tree_path", str(vocab_tree_path),
        ]
        print(f"Running COLMAP sequential feature matching (overlap {overlap}, vocab tree loop detection)...")
    else:
        print(f"Running COLMAP sequential feature matching (overlap {overlap}, window only)...")

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

def run_arkit_pose_triangulation(frames, database_path, image_path, seed_dir, output_dir):
    """Reconstruct by triangulating landmarks against fixed ARKit poses.

    Vision-only SfM (incremental or GLOMAP) fails to cohere on wide-baseline /
    low-parallax capture (e.g. park foliage) even with rich matches. Instead of
    estimating poses from images, seed a COLMAP model with the trusted ARKit poses
    for every frame and run point_triangulator to place landmarks. This yields a
    fully-connected reconstruction over all posed frames -- poses never depend on
    the visual view graph.
    """
    seed_dir = Path(seed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n = create_arkit_seed_model(frames, database_path, seed_dir)
    if n == 0:
        print("ARKit seed model is empty (no frames matched database images)")
        return False

    cmd = [
        "colmap", "point_triangulator",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--input_path", str(seed_dir),
        "--output_path", str(output_path),
        # Trust ARKit calibration + poses; don't let BA drift intrinsics.
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        # Looser triangulation thresholds recover more/longer tracks: ARKit VIO
        # drift otherwise splits multi-view observations into separate 2-view
        # points that later fail the >=3-sightings cull. Merging them into longer
        # tracks roughly doubles the usable-landmark pool; the refiner's bundle
        # adjustment + outlier removal then prunes any bad merges.
        "--Mapper.tri_complete_max_reproj_error", "12",
        "--Mapper.tri_merge_max_reproj_error", "12",
        "--Mapper.filter_max_reproj_error", "12",
        "--Mapper.tri_min_angle", "1.0",
    ]

    print("Running ARKit-pose triangulation (point_triangulator)...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print("Triangulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Triangulation failed: {e}")
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
    if args.use_sequential:
        print("\nRunning sequential feature matching...")
        vocab_tree_path = resolve_vocab_tree_path(args.source_dir)
        if vocab_tree_path is None:
            print("No vocab tree found, running window-only sequential matching (no loop detection)")
        else:
            print(f"Using vocab tree: {vocab_tree_path}")
        if not run_colmap_sequential_feature_matching(database_path, args.sequential_overlap, vocab_tree_path):
            print("COLMAP pipeline failed at sequential feature matching")
            exit(1)
    elif args.use_vocab_tree:
        print("\nRunning vocab tree feature matching...")
        vocab_tree_path = resolve_vocab_tree_path(args.source_dir)
        if vocab_tree_path is None:
            print("No vocab tree found (looked in source_dir and next to script)")
            exit(1)
        print(f"Using vocab tree: {vocab_tree_path}")
        if not run_colmap_vocab_tree_feature_matching(database_path, vocab_tree_path):
            print("COLMAP pipeline failed at vocab tree feature matching")
            exit(1)
    else:
        print("\nRunning exhaustive feature matching...")
        if not run_colmap_feature_matching(database_path):
            print("COLMAP pipeline failed at feature matching")
            exit(1)

def sparse_reconstruction(args, arkit_frames, database_path, sparse_dir):
    if args.use_arkit_poses:
        print("\nReconstructing by triangulating against ARKit poses...")
        work_dir = Path(database_path).parent
        seed_dir = work_dir / "arkit_seed"
        # point_triangulator writes the model directly into output_path; use
        # sparse/0 so the rest of the pipeline (which expects sparse/0) works.
        if not run_arkit_pose_triangulation(arkit_frames, database_path, work_dir, seed_dir, sparse_dir / "0"):
            print("COLMAP pipeline failed at ARKit-pose triangulation")
            exit(1)
    elif args.use_glomap:
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
    parser.add_argument("--use_sequential", action="store_true",
                       help="Use sequential (sliding-window) matching, with vocab tree loop detection. "
                            "Uses <source_dir>/vocab_tree.bin if present, else the bundled default. "
                            "Best for sequential captures.")
    parser.add_argument("--sequential_overlap", type=int, default=10,
                       help="Number of subsequent images to match per image for sequential matching (default: 10)")
    parser.add_argument("--use_glomap", action="store_true",
                       help="Use GLOMAP for reconstruction instead of COLMAP (faster, global SfM)")
    parser.add_argument("--use_arkit_poses", action="store_true",
                       help="Reconstruct by triangulating landmarks against fixed ARKit poses "
                            "(seed model + point_triangulator). Best when vision-only SfM cannot "
                            "cohere (wide-baseline / low-parallax capture). Produces a fully "
                            "connected model in ARKit coordinates; skips model alignment. Takes "
                            "precedence over the other reconstruction options.")

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

    # Step 4: Sparse reconstruction
    reconstruction_path = sparse_reconstruction(args, arkit_frames, database_path, sparse_dir)
    reconstruction_path = sparse_dir / "0"

    # Step 6: Model alignment. Skipped for --use_arkit_poses: the model is already
    # built directly in ARKit world coordinates, so there is nothing to align.
    if args.use_arkit_poses:
        print("\nSkipping model alignment (ARKit-pose model is already in ARKit coordinates)")
    else:
        reconstruction_path = model_alignment(args, arkit_frames, ref_coords_file, reconstruction_path, database_path)

    # Step 7: Export map
    export_map(args, database_path, map_json_file, arkit_frames, reconstruction_path, poses_dir)

    print(f"\n✅ Processing completed successfully!")
    print(f"Feature extraction method: {'COLMAP' if args.use_colmap_sift else 'OpenCV'}")
    matching_method = "Sequential" if args.use_sequential else ("Vocabulary Tree" if args.use_vocab_tree else "Exhaustive")
    print(f"Matching method: {matching_method}")
    if args.use_arkit_poses:
        reconstruction_method = "ARKit-pose triangulation"
    elif args.use_glomap:
        reconstruction_method = "GLOMAP"
    else:
        reconstruction_method = "COLMAP"
    print(f"Reconstruction method: {reconstruction_method}")
    print(f"Model alignment: {'Skipped (ARKit coords)' if args.use_arkit_poses else 'Applied in place'}")
    print(f"Final map: {map_json_file}")
    print(f"Launch gui with: colmap gui --database_path {database_path} --import_path {reconstruction_path} --image_path {work_dir}")
    return 0

if __name__ == "__main__":
    exit(main())