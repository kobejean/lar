import numpy as np
from pathlib import Path
import subprocess
import struct
import sqlite3
from colmap_pose import ColmapPose

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
        pos_std = 1.0
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