#!/usr/bin/env python3
"""
Utility script for validating and visualizing pose priors in COLMAP database
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
import argparse

def inspect_pose_priors(database_path):
    """Inspect pose priors stored in COLMAP database"""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Check if pose_priors table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'")
    if not cursor.fetchone():
        print("No pose_priors table found in database")
        return False
    
    # Get all pose priors with image names (correct COLMAP schema)
    cursor.execute('''
        SELECT i.name, i.prior_qw, i.prior_qx, i.prior_qy, i.prior_qz,
               i.prior_tx, i.prior_ty, i.prior_tz,
               p.position, p.coordinate_system, p.position_covariance
        FROM images i
        JOIN pose_priors p ON i.image_id = p.image_id
        ORDER BY i.name
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No pose priors found")
        return False
    
    print(f"Found {len(results)} pose priors:")
    print("=" * 80)
    
    positions = []
    for row in results:
        name, qw, qx, qy, qz, tx, ty, tz, position_blob, coord_sys, covariance_blob = row
        
        # Convert world-to-camera to camera-to-world for visualization
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        camera_position = -R.T @ t
        positions.append(camera_position)
        
        # Unpack position from binary blob
        import struct
        px, py, pz = struct.unpack('ddd', position_blob)
        
        # Unpack covariance matrix from binary blob
        cov_flat = struct.unpack('ddddddddd', covariance_blob)
        cov_matrix = np.array(cov_flat).reshape(3, 3)
        
        print(f"Image: {name}")
        print(f"  Position (world): [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
        print(f"  Position (prior): [{px:.3f}, {py:.3f}, {pz:.3f}]")
        print(f"  Quaternion (w2c): [{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]")
        print(f"  Coordinate system: {coord_sys} {'(Cartesian)' if coord_sys == -1 else '(GPS)' if coord_sys == 0 else '(Unknown)'}")
        print(f"  Covariance diagonal: [{cov_matrix[0,0]:.1f}, {cov_matrix[1,1]:.1f}, {cov_matrix[2,2]:.1f}]")
        print()
    
    # Calculate trajectory statistics
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    
    print("Trajectory Statistics:")
    print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Mean distance from centroid: {np.mean(distances):.3f}")
    print(f"  Max distance from centroid: {np.max(distances):.3f}")
    print(f"  Position range X: {np.min(positions[:, 0]):.3f} to {np.max(positions[:, 0]):.3f}")
    print(f"  Position range Y: {np.min(positions[:, 1]):.3f} to {np.max(positions[:, 1]):.3f}")
    print(f"  Position range Z: {np.min(positions[:, 2]):.3f} to {np.max(positions[:, 2]):.3f}")
    
    return True

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

def compare_with_arkit(database_path, frames_json_path):
    """Compare database pose priors with original ARKit poses"""
    # Load ARKit data
    with open(frames_json_path, 'r') as f:
        frames = json.load(f)
    
    arkit_poses = {}
    for frame in frames:
        image_name = f"{frame['id']:08d}_image.jpeg"
        matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
        camera_position = matrix[:3, 3]
        arkit_poses[image_name] = camera_position
    
    # Load database poses
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz
        FROM images
        WHERE prior_qw IS NOT NULL
    ''')
    
    db_poses = {}
    for row in cursor.fetchall():
        name, qw, qx, qy, qz, tx, ty, tz = row
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        camera_position = -R.T @ t
        db_poses[name] = camera_position
    
    conn.close()
    
    # Compare poses
    print("Comparing ARKit vs Database poses:")
    print("=" * 60)
    
    total_error = 0
    count = 0
    
    for image_name in sorted(arkit_poses.keys()):
        if image_name in db_poses:
            arkit_pos = arkit_poses[image_name]
            db_pos = db_poses[image_name]
            error = np.linalg.norm(arkit_pos - db_pos)
            
            print(f"{image_name}:")
            print(f"  ARKit: [{arkit_pos[0]:.3f}, {arkit_pos[1]:.3f}, {arkit_pos[2]:.3f}]")
            print(f"  DB:    [{db_pos[0]:.3f}, {db_pos[1]:.3f}, {db_pos[2]:.3f}]")
            print(f"  Error: {error:.6f}")
            print()
            
            total_error += error
            count += 1
    
    if count > 0:
        avg_error = total_error / count
        print(f"Average position error: {avg_error:.6f}")
        if avg_error < 1e-6:
            print("✅ Poses match perfectly!")
        elif avg_error < 1e-3:
            print("✅ Poses match with minimal numerical error")
        else:
            print("⚠️  Significant pose differences detected")

def suggest_variance_parameters(frames_json_path):
    """Analyze ARKit trajectory to suggest appropriate variance parameters"""
    with open(frames_json_path, 'r') as f:
        frames = json.load(f)
    
    positions = []
    for frame in frames:
        matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
        camera_position = matrix[:3, 3]
        positions.append(camera_position)
    
    positions = np.array(positions)
    
    # Calculate inter-frame distances
    distances = []
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[i-1])
        distances.append(dist)
    
    distances = np.array(distances)
    
    print("ARKit Trajectory Analysis:")
    print("=" * 40)
    print(f"Number of frames: {len(frames)}")
    print(f"Total trajectory length: {np.sum(distances):.3f}")
    print(f"Average inter-frame distance: {np.mean(distances):.3f}")
    print(f"Max inter-frame distance: {np.max(distances):.3f}")
    print(f"Min inter-frame distance: {np.min(distances):.3f}")
    
    # Suggest variance based on trajectory characteristics
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    
    # Conservative: variance should be smaller than typical movements
    conservative_var = (avg_distance * 0.5) ** 2
    # Moderate: variance based on average movement
    moderate_var = avg_distance ** 2
    # Liberal: variance based on maximum movement
    liberal_var = (max_distance * 0.8) ** 2
    
    print(f"\nSuggested pose variance parameters:")
    print(f"  Conservative (tight priors): {conservative_var:.1f}")
    print(f"  Moderate (balanced):         {moderate_var:.1f}")
    print(f"  Liberal (loose priors):      {liberal_var:.1f}")
    print(f"\nRecommendation: Start with moderate ({moderate_var:.1f}) and adjust based on results")

def main():
    parser = argparse.ArgumentParser(description="Utility for pose prior validation")
    parser.add_argument("command", choices=["inspect", "compare", "suggest"],
                       help="Command to run")
    parser.add_argument("--database", help="Path to COLMAP database")
    parser.add_argument("--frames_json", help="Path to ARKit frames.json")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        if not args.database:
            print("Error: --database required for inspect command")
            return 1
        inspect_pose_priors(args.database)
    
    elif args.command == "compare":
        if not args.database or not args.frames_json:
            print("Error: --database and --frames_json required for compare command")
            return 1
        compare_with_arkit(args.database, args.frames_json)
    
    elif args.command == "suggest":
        if not args.frames_json:
            print("Error: --frames_json required for suggest command")
            return 1
        suggest_variance_parameters(args.frames_json)
    
    return 0

if __name__ == "__main__":
    exit(main())