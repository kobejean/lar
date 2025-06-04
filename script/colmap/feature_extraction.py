import subprocess
import numpy as np
import cv2
import base64
import sqlite3
from pathlib import Path

def extract_colmap_sift_features(work_dir, database_path, max_num_features=8192):
    """Extract SIFT features using COLMAP's built-in feature extractor and export to text files"""
    
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(work_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "0",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--SiftExtraction.first_octave", "-1",
        "--SiftExtraction.num_octaves", "4",
        "--SiftExtraction.octave_resolution", "3",
        "--SiftExtraction.peak_threshold", "0.00666667",
        "--SiftExtraction.edge_threshold", "10.0"
    ]
    
    print(f"Running COLMAP feature extraction (max {max_num_features} features per image)...")
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        print("COLMAP feature extraction completed successfully")
        
        # Export features from database to text files
        if not export_colmap_features_to_text(database_path, work_dir):
            print("Failed to export COLMAP features to text files")
            return False
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"COLMAP feature extraction failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def export_colmap_features_to_text(database_path, work_dir):
    """Export COLMAP features from database to text files"""
    print("Exporting COLMAP features to text files...")
    
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get all images with their IDs and names
        cursor.execute("SELECT image_id, name FROM images")
        images = cursor.fetchall()
        
        total_features = 0
        
        for image_id, image_name in images:
            # Get keypoints
            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
            keypoints_row = cursor.fetchone()
            
            # Get descriptors
            cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,))
            descriptors_row = cursor.fetchone()
            
            if keypoints_row and descriptors_row:
                # Parse keypoints
                kp_rows, kp_cols, kp_data = keypoints_row
                keypoints_data = np.frombuffer(kp_data, dtype=np.float32)
                
                # COLMAP keypoints format: each keypoint has 6 values [x, y, a11, a12, a21, a22]
                # where a11, a12, a21, a22 represent the affine transformation matrix
                if kp_cols == 6:  # Standard COLMAP format
                    keypoints_reshaped = keypoints_data.reshape(kp_rows, kp_cols)
                    # Convert to [x, y, scale, orientation] format
                    keypoints_converted = []
                    for kp in keypoints_reshaped:
                        x, y = kp[0], kp[1]
                        # Extract scale and orientation from affine matrix
                        a11, a12, a21, a22 = kp[2], kp[3], kp[4], kp[5]
                        # Scale is roughly the determinant of the affine matrix
                        scale = np.sqrt(abs(a11 * a22 - a12 * a21))
                        # Orientation from the affine matrix
                        orientation = np.degrees(np.arctan2(a21, a11))
                        keypoints_converted.append([x, y, scale, orientation])
                    keypoints_array = np.array(keypoints_converted)
                elif kp_cols == 4:  # Already in [x, y, scale, orientation] format
                    keypoints_array = keypoints_data.reshape(kp_rows, kp_cols)
                else:
                    print(f"Warning: Unexpected keypoint format for {image_name}: {kp_rows}x{kp_cols}")
                    continue
                
                # Parse descriptors
                desc_rows, desc_cols, desc_data = descriptors_row
                descriptors_data = np.frombuffer(desc_data, dtype=np.uint8)
                descriptors_array = descriptors_data.reshape(desc_rows, desc_cols)
                
                # Verify dimensions match
                if len(keypoints_array) != len(descriptors_array):
                    print(f"Warning: Keypoint/descriptor count mismatch for {image_name}: {len(keypoints_array)} vs {len(descriptors_array)}")
                    min_count = min(len(keypoints_array), len(descriptors_array))
                    keypoints_array = keypoints_array[:min_count]
                    descriptors_array = descriptors_array[:min_count]
                
                # Write features to text file in COLMAP format
                image_path = Path(work_dir) / image_name
                feature_file = image_path.with_suffix(image_path.suffix + '.txt')
                write_colmap_features_from_arrays(keypoints_array, descriptors_array, feature_file)
                
                total_features += len(keypoints_array)
                print(f"  Exported {len(keypoints_array)} features for {image_name} (kp: {kp_rows}x{kp_cols}, desc: {desc_rows}x{desc_cols})")
        
        conn.close()
        
        print(f"Total features exported: {total_features}")
        return True
        
    except Exception as e:
        print(f"Error exporting COLMAP features to text: {e}")
        import traceback
        traceback.print_exc()
        return False

def write_colmap_features_from_arrays(keypoints_array, descriptors_array, output_file):
    """Write keypoints and descriptors arrays to COLMAP text format"""
    if descriptors_array is None or len(keypoints_array) == 0:
        print(f"Warning: No features to write for {output_file}")
        return
    
    with open(output_file, 'w') as f:
        # Header: NUM_FEATURES DESCRIPTOR_SIZE
        f.write(f"{len(keypoints_array)} 128\n")
        
        # Write each feature
        for kp, desc in zip(keypoints_array, descriptors_array):
            # COLMAP format: X Y SCALE ORIENTATION D_1 D_2 ... D_128
            x, y, scale, orientation = kp[0], kp[1], kp[2], kp[3]
            desc_str = ' '.join(str(int(d)) for d in desc)
            
            f.write(f"{x:.6f} {y:.6f} {scale:.6f} {orientation:.6f} {desc_str}\n")

def extract_opencv_sift_features(work_dir, output_features=True, max_num_features=8192):
    """Extract SIFT features using OpenCV, filter by scale to keep largest-scale features"""
    work_path = Path(work_dir)
    image_files = list(work_path.glob("*_image.jpeg"))
    print(f"Extracting SIFT features from {len(image_files)} images using OpenCV...")
    
    # Initialize SIFT detector with no feature limit (extract all features first)
    sift = cv2.SIFT_create(
        nfeatures=0,  # Extract all features initially
        nOctaveLayers=3,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6,
        descriptorType=cv2.CV_8U
    )
    
    features_extracted = 0
    
    for img_file in image_files:
        print(f"Processing {img_file.name}...")
        
        # Read image
        try:
            image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image {img_file}")
                continue
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            continue
        
        # Detect and compute SIFT features
        try:
            keypoints, descriptors = sift.detectAndCompute(image, None)
            
            if keypoints is None or descriptors is None or len(keypoints) == 0:
                print(f"  No features found")
                continue
            
            print(f"  Found {len(keypoints)} raw features")
            
            # Filter by scale to keep the largest-scale (most prominent) features
            if len(keypoints) > max_num_features:
                # Extract scales (size property in OpenCV keypoints)
                scales = np.array([kp.size for kp in keypoints])
                
                # Get indices of features sorted by scale (descending)
                scale_indices = np.argsort(scales)[::-1]
                
                # Keep only the top max_num_features by scale
                top_indices = scale_indices[:max_num_features]
                
                # Filter keypoints and descriptors
                filtered_keypoints = [keypoints[i] for i in top_indices]
                filtered_descriptors = descriptors[top_indices]
                
                print(f"  Filtered to {len(filtered_keypoints)} largest-scale features (scale range: {scales[top_indices].max():.2f} - {scales[top_indices].min():.2f})")
            else:
                filtered_keypoints = keypoints
                filtered_descriptors = descriptors
                print(f"  Kept all {len(filtered_keypoints)} features")
            
            features_extracted += len(filtered_keypoints)
            
            if output_features:
                # Write features to text file in COLMAP format
                feature_file = img_file.with_suffix(img_file.suffix + '.txt')
                write_colmap_features(filtered_keypoints, filtered_descriptors, feature_file)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"Total features extracted: {features_extracted}")
    return True

def write_colmap_features(keypoints, descriptors, output_file):
    """Write keypoints and descriptors in COLMAP text format with proper validation"""
    if descriptors is None or len(keypoints) == 0:
        print(f"Warning: No features to write for {output_file}")
        return
    
    with open(output_file, 'w') as f:
        # Header: NUM_FEATURES DESCRIPTOR_SIZE
        f.write(f"{len(keypoints)} 128\n")
        
        # Write each feature
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            # COLMAP format: X Y SCALE ORIENTATION D_1 D_2 ... D_128
            x, y = kp.pt
            scale = kp.size
            orientation = kp.angle
            if descriptors.dtype != np.uint8:
              # Proper normalization to [0, 255] range
              descriptors = (descriptors / np.max(descriptors) * 255).astype(np.uint8)
            desc_str = ' '.join(str(int(d)) for d in desc)
            
            f.write(f"{x:.6f} {y:.6f} {scale:.6f} {orientation:.6f} {desc_str}\n")

def get_descriptor_for_3d_point(track, colmap_images, database_path):
    """Get the SIFT descriptor for a 3D point by reading directly from COLMAP database"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        for img_id, point2d_idx in track:
            if img_id in colmap_images:
                # Get descriptors for this image
                cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (img_id,))
                descriptors_row = cursor.fetchone()
                
                if descriptors_row:
                    desc_rows, desc_cols, desc_data = descriptors_row
                    descriptors_data = np.frombuffer(desc_data, dtype=np.uint8)
                    descriptors_array = descriptors_data.reshape(desc_rows, desc_cols)
                    
                    if point2d_idx < len(descriptors_array):
                        descriptor = descriptors_array[point2d_idx]
                        descriptor_b64 = base64.b64encode(descriptor.tobytes()).decode('ascii')
                        conn.close()
                        return descriptor_b64
        
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error reading descriptor from database: {e}")
        return None
