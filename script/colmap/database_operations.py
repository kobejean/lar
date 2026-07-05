import numpy as np
from pathlib import Path
import subprocess
import struct
import sqlite3
from colmap_pose import (
    ColmapPose,
    rotation_matrix_to_quaternion,
    extract_rotation_translation_from_extrinsics,
)

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

_COLMAP_CAMERA_MODEL_NAME = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL"}

def create_arkit_seed_model(frames, database_path, seed_dir):
    """Write a COLMAP text model seeded with ARKit poses for point_triangulator.

    Uses the (corrected) ARKit->COLMAP world-to-camera convention from
    extract_rotation_translation_from_extrinsics. Cameras and image ids/names are
    taken from the database so they line up with the extracted features. Points3D
    is left empty; point_triangulator fills it using the fixed poses. Only frames
    present in the database are written (e.g. tail frames with no ARKit pose or
    removed from the db are skipped).

    Returns the number of images written.
    """
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()
    db_images = {name: (image_id, camera_id)
                 for image_id, name, camera_id in cur.execute("SELECT image_id, name, camera_id FROM images")}
    db_cameras = {}
    for camera_id, model, w, h, params in cur.execute("SELECT camera_id, model, width, height, params FROM cameras"):
        db_cameras[camera_id] = (model, w, h, struct.unpack(f"{len(params)//8}d", params))
    conn.close()

    with open(seed_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, (model, w, h, params) in sorted(db_cameras.items()):
            model_name = _COLMAP_CAMERA_MODEL_NAME.get(model, model)
            f.write(f"{cid} {model_name} {w} {h} " + " ".join(f"{p:.12f}" for p in params) + "\n")

    written = 0
    with open(seed_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] (empty; filled by triangulator)\n")
        for frame in frames:
            name = f"{frame['id']:08d}_image.jpeg"
            if name not in db_images:
                continue
            image_id, camera_id = db_images[name]
            R_w2c, t_w2c = extract_rotation_translation_from_extrinsics(
                frame["extrinsics"], apply_colmap_conversion=True)
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
            f.write(f"{image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                    f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} {camera_id} {name}\n\n")
            written += 1

    with open(seed_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list (empty; filled by point_triangulator)\n")

    print(f"Created ARKit seed model with {written} images at {seed_dir}")
    return written
