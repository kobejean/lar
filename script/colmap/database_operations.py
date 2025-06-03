import numpy as np
from pathlib import Path
import subprocess
import struct

def create_colmap_database_with_pose_priors(frames, database_path, work_dir, pose_variance=25.0):
    """Create COLMAP database with ARKit pose priors for spatial matching"""
    import sqlite3
    
    # Ensure the directory exists
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
    
    # Create database and tables
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        )
    ''')
    
    # Create images table  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
    ''')
    
    # Create keypoints and descriptors tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    # PINHOLE model ID in COLMAP
    PINHOLE_MODEL_ID = 1
    
    # Group frames by unique camera parameters to reduce redundant cameras
    unique_cameras = {}
    camera_mapping = {}
    
    for frame in frames:
        intrinsics = frame['intrinsics']
        fx, fy = intrinsics[0], intrinsics[4]
        cx, cy = intrinsics[6], intrinsics[7]
        
        # Create a key for unique camera parameters
        camera_key = (round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
        
        if camera_key not in unique_cameras:
            # Pack parameters as binary data (COLMAP format)
            params_blob = struct.pack('dddd', fx, fy, cx, cy)
            
            cursor.execute('''
                INSERT INTO cameras (model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?)
            ''', (PINHOLE_MODEL_ID, 1920, 1440, params_blob, 1))
            
            camera_id = cursor.lastrowid
            unique_cameras[camera_key] = camera_id
            print(f"Camera {camera_id}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        camera_mapping[frame['id']] = unique_cameras[camera_key]
    
    # Insert image entries with pose priors
    for frame in frames:
        image_name = f"{frame['id']:08d}_image.jpeg"
        camera_id = camera_mapping[frame['id']]
        
        # Extract pose from ARKit extrinsics (camera-to-world transformation)
        matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
        
        # ARKit provides camera-to-world transformation
        # Extract translation (camera position in world coordinates)
        translation = matrix[:3, 3]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = matrix[:3, :3]
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        
        # COLMAP expects world-to-camera transformation for priors
        # Convert camera-to-world to world-to-camera
        R_world_to_cam = rotation_matrix.T
        t_world_to_cam = -R_world_to_cam @ translation
        q_world_to_cam = rotation_matrix_to_quaternion(R_world_to_cam)
        
        cursor.execute('''
            INSERT INTO images (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, 
                              prior_tx, prior_ty, prior_tz)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_name, camera_id, 
              float(q_world_to_cam[0]), float(q_world_to_cam[1]), 
              float(q_world_to_cam[2]), float(q_world_to_cam[3]),
              float(t_world_to_cam[0]), float(t_world_to_cam[1]), float(t_world_to_cam[2])))
        
        print(f"Added image {image_name} with camera {camera_id} and pose prior")
        print(f"  Position: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
    
    # Create pose_priors table for spatial matching (correct COLMAP schema)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB NOT NULL,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    # Add pose priors with position and covariance
    cursor.execute("SELECT image_id FROM images")
    image_ids = cursor.fetchall()
    
    # Get the camera positions for each image
    for (image_id,) in image_ids:
        # Get the prior pose for this image
        cursor.execute('''
            SELECT prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz
            FROM images WHERE image_id = ?
        ''', (image_id,))
        
        row = cursor.fetchone()
        if row:
            qw, qx, qy, qz, tx, ty, tz = row
            
            # Convert world-to-camera back to camera position for pose priors
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            camera_position = -R.T @ t
            
            # Pack position as binary blob (3 float64 values)
            position_blob = struct.pack('ddd', camera_position[0], camera_position[1], camera_position[2])
            
            # Create 3x3 covariance matrix (diagonal with specified variance)
            covariance_matrix = np.eye(3) * pose_variance
            # Pack covariance as binary blob (9 float64 values in row-major order)
            covariance_blob = struct.pack('ddddddddd', *covariance_matrix.flatten())
            
            # Insert pose prior: coordinate_system = 2 means CARTESIAN (from PyColmap: CARTESIAN = 2)
            cursor.execute('''
                INSERT INTO pose_priors 
                (image_id, position, coordinate_system, position_covariance)
                VALUES (?, ?, ?, ?)
            ''', (image_id, position_blob, 2, covariance_blob))
    
    conn.commit()
    conn.close()
    
    print(f"Created COLMAP database with {len(unique_cameras)} unique cameras, {len(frames)} images, and pose priors")
    print(f"Pose variance: {pose_variance} (position), 0.1 (orientation)")

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

def create_colmap_database(frames, database_path, work_dir):
    """Create COLMAP database with proper camera setup for feature import"""
    import sqlite3
    
    # Ensure the directory exists
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
    
    # Create database and tables
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        )
    ''')
    
    # Create images table  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
    ''')
    
    # Create keypoints and descriptors tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    # PINHOLE model ID in COLMAP
    PINHOLE_MODEL_ID = 1
    
    # Group frames by unique camera parameters to reduce redundant cameras
    unique_cameras = {}
    camera_mapping = {}
    
    for frame in frames:
        intrinsics = frame['intrinsics']
        fx, fy = intrinsics[0], intrinsics[4]
        cx, cy = intrinsics[6], intrinsics[7]
        
        # Create a key for unique camera parameters
        camera_key = (round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
        
        if camera_key not in unique_cameras:
            # Pack parameters as binary data (COLMAP format)
            params_blob = struct.pack('dddd', fx, fy, cx, cy)
            
            cursor.execute('''
                INSERT INTO cameras (model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?)
            ''', (PINHOLE_MODEL_ID, 1920, 1440, params_blob, 1))
            
            camera_id = cursor.lastrowid
            unique_cameras[camera_key] = camera_id
            print(f"Camera {camera_id}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        camera_mapping[frame['id']] = unique_cameras[camera_key]
    
    # Insert image entries
    for frame in frames:
        image_name = f"{frame['id']:08d}_image.jpeg"
        camera_id = camera_mapping[frame['id']]
        
        cursor.execute('''
            INSERT INTO images (name, camera_id)
            VALUES (?, ?)
        ''', (image_name, camera_id))
        
        print(f"Added image {image_name} with camera {camera_id}")
    
    conn.commit()
    conn.close()
    
    print(f"Created COLMAP database with {len(unique_cameras)} unique cameras and {len(frames)} images")

def create_empty_colmap_database(work_dir):
    """Create an empty COLMAP database for feature extraction"""
    import sqlite3
    
    database_path = work_dir / "database.db"
    
    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
    
    # Create database and tables
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Create cameras table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        )
    ''')
    
    # Create images table  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
    ''')
    
    # Create keypoints and descriptors tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Created empty COLMAP database at {database_path}")
    return database_path

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def read_colmap_poses(images_txt_path):
    """Read COLMAP poses from images.txt and return camera positions"""
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
                    image_id = int(parts[0])
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    camera_id = int(parts[8])
                    image_name = parts[9]
                    
                    # Convert quaternion to rotation matrix
                    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                    t = np.array([tx, ty, tz])
                    
                    # Camera position in world coordinates: -R^T * t
                    camera_position = -R.T @ t
                    
                    poses[image_name] = {
                        'position': camera_position,
                        'image_id': image_id,
                        'camera_id': camera_id
                    }
                    
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