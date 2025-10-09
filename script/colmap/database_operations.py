import numpy as np
from pathlib import Path
import subprocess
import struct
import sqlite3
from colmap_pose import ColmapPose, rotation_matrix_to_quaternion

def create_colmap_database(frames, database_path, work_dir):
    """Create COLMAP database with proper camera setup for feature import"""
    
    # Ensure the directory exists
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
    
    # Step 1: Create empty database using COLMAP's official command
    print("Creating COLMAP database...")
    cmd = ["colmap", "database_creator", "--database_path", str(database_path)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created COLMAP database at {database_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create COLMAP database: {e}")
        print(f"stderr: {e.stderr}")
        return False
    
    # Step 2: Populate the database with our data
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # PINHOLE model ID in COLMAP
    PINHOLE_MODEL_ID = 1
    
    for frame in frames:
        intrinsics = frame['intrinsics']
        fx, fy = intrinsics[0], intrinsics[4]
        cx, cy = intrinsics[6], intrinsics[7]
        
        # Pack parameters as binary data (COLMAP format)
        params_blob = struct.pack('dddd', fx, fy, cx, cy)
        
        cursor.execute('''
            INSERT INTO cameras (model, width, height, params, prior_focal_length)
            VALUES (?, ?, ?, ?, ?)
        ''', (PINHOLE_MODEL_ID, 1920, 1440, params_blob, 1))
    
        # Insert image entries
        image_name = f"{frame['id']:08d}_image.jpeg"
        camera_id = frame['id']+1
        
        cursor.execute('''
            INSERT INTO images (name, camera_id)
            VALUES (?, ?)
        ''', (image_name, camera_id))
        
        print(f"Added image {image_name} with camera {camera_id}")
        
        # Add pose prior
        extrinsics = frame['extrinsics']
        x, y, z = extrinsics[12], extrinsics[13], extrinsics[14]
        position_blob = struct.pack('ddd', x, -y, -z)
        pos_std = 10.0
        covariance_blob = struct.pack('ddddddddd',
            pos_std, 0, 0,
            0, pos_std, 0,
            0, 0, pos_std
        )
        image_id = frame['id']+1
        cursor.execute('''
            INSERT INTO pose_priors (image_id, position, coordinate_system, position_covariance)
            VALUES (?, ?, ?, ?)
        ''', (image_id, position_blob, 1, covariance_blob))
    
    conn.commit()
    conn.close()
    
    print(f"Created COLMAP database with {len(frames)} images")
    return True

def read_colmap_poses(images_txt_path):
    """
    Read COLMAP poses from images.txt
    
    Args:
        images_txt_path: Path to COLMAP images.txt file
    
    Returns:
        Dictionary mapping image_name to ColmapPose object
    """
    poses = {}
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            
            # Parse image info line
            parts = line.split()
            if len(parts) >= 10:
                try:
                    pose = ColmapPose(parts)
                    poses[pose.image_id] = pose

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}")
            
            i += 2  # Skip the 2D points line
    
    print(f"Successfully read {len(poses)} COLMAP poses")
    return poses

def export_poses(sparse_dir, output_dir):
    """Export camera poses to a readable format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(output_path),
        "--output_type", "TXT"
    ]

    print("Exporting camera poses...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"Camera poses exported to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Pose export failed: {e}")
        return False

# ============================================================================
# Two-View Geometry Operations for ARKit Integration
# ============================================================================

def image_ids_to_pair_id(image_id1, image_id2):
    """
    Convert two image IDs to a unique pair ID for COLMAP database.

    Args:
        image_id1, image_id2: Image IDs (1-indexed)

    Returns:
        Unique pair ID (row-major index in upper-triangular match matrix)
    """
    MAX_IMAGE_ID = 2147483647  # Maximum signed 32-bit integer
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def array_to_blob(array):
    """Convert numpy array to binary blob in float64 format for COLMAP database."""
    return array.astype(np.float64).tobytes()

def extract_pose_from_extrinsics(extrinsics):
    """
    Extract rotation matrix and translation vector from ARKit extrinsics.

    ARKit extrinsics is a 16-element array representing a 4x4 transformation
    matrix in column-major order (camera-from-world transform).

    Args:
        extrinsics: 16-element array [R00, R10, R20, 0, R01, R11, R21, 0,
                                       R02, R12, R22, 0, tx, ty, tz, 1]

    Returns:
        (R, t): 3x3 rotation matrix and 3-element translation vector
    """
    R = np.array([
        [extrinsics[0], extrinsics[4], extrinsics[8]],
        [extrinsics[1], extrinsics[5], extrinsics[9]],
        [extrinsics[2], extrinsics[6], extrinsics[10]]
    ])
    t = np.array([extrinsics[12], extrinsics[13], extrinsics[14]])
    return R, t

def compute_relative_pose(extrinsics1, extrinsics2):
    """
    Compute relative pose from camera1 to camera2.

    Given two camera-from-world transforms T1 and T2, compute the
    camera2-from-camera1 transform: T_rel = T2 * T1^-1

    Args:
        extrinsics1: ARKit extrinsics for camera 1
        extrinsics2: ARKit extrinsics for camera 2

    Returns:
        (R_rel, t_rel): Relative rotation matrix and translation vector
    """
    # Extract poses
    R1, t1 = extract_pose_from_extrinsics(extrinsics1)
    R2, t2 = extract_pose_from_extrinsics(extrinsics2)

    # Compute world-from-camera1 (invert T1)
    R1_inv = R1.T
    t1_inv = -R1.T @ t1

    # Compute camera2-from-camera1: T2 * T1^-1
    R_rel = R2 @ R1_inv
    t_rel = R2 @ t1_inv + t2

    return R_rel, t_rel

def insert_two_view_geometries_from_arkit(frames, database_path, use_empty_matches=True):
    """
    Insert ARKit relative poses into COLMAP's two_view_geometries table.

    This provides high-quality odometry constraints for global bundle adjustment
    by inserting relative poses between sequential frames. These constraints help
    COLMAP/GLOMAP optimize the reconstruction using ARKit's accurate VIO tracking.

    The qvec and tvec are the key data that bundle adjustment uses. The F, E, H
    matrices are set to identity as they're not needed for pose-only constraints.

    Args:
        frames: List of ARKit frame dictionaries with 'id' and 'extrinsics'
        database_path: Path to COLMAP database
        use_empty_matches: If True, insert minimal placeholder matches (recommended)

    Returns:
        Number of two-view geometries inserted
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # CALIBRATED config type (we have known intrinsics and metric poses)
    CONFIG_CALIBRATED = 2

    count = 0
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # Image IDs in COLMAP (1-indexed, matching what we insert in create_colmap_database)
        image_id1 = frame1['id'] + 1
        image_id2 = frame2['id'] + 1

        # Compute relative pose from frame1 to frame2
        R_rel, t_rel = compute_relative_pose(frame1['extrinsics'], frame2['extrinsics'])

        # Convert rotation to quaternion (wxyz format) using existing function
        qvec = rotation_matrix_to_quaternion(R_rel)

        # Minimal placeholder matches (bundle adjustment doesn't need feature correspondences)
        matches = np.array([[0, 0]], dtype=np.uint32) if use_empty_matches else np.array([], dtype=np.uint32).reshape(0, 2)

        # Compute pair_id
        pair_id = image_ids_to_pair_id(image_id1, image_id2)

        # Use identity matrices for F, E, H (not used by bundle adjustment)
        F = np.eye(3)
        E = np.eye(3)
        H = np.eye(3)

        # Insert into two_view_geometries table
        cursor.execute('''
            INSERT INTO two_view_geometries
            (pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(pair_id),
            matches.shape[0],
            matches.shape[1],
            matches.tobytes(),
            CONFIG_CALIBRATED,
            array_to_blob(F),
            array_to_blob(E),
            array_to_blob(H),
            array_to_blob(qvec),
            array_to_blob(t_rel)
        ))

        count += 1
        if (count % 100) == 0:
            print(f"Inserted {count} two-view geometries...")

    conn.commit()
    conn.close()

    print(f"Successfully inserted {count} ARKit relative poses into two_view_geometries")
    return count