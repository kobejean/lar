import json
import numpy as np
from database_operations import read_colmap_poses
from feature_extraction import get_descriptor_for_3d_point

def calculate_spatial_bounds(landmark_position, camera_positions, max_distance_factor=2.0):
    """Calculate spatial bounds for a landmark based on camera positions that observe it"""
    if not camera_positions:
        return {
            "lower": {"x": landmark_position[0] - 1.0, "y": landmark_position[2] - 1.0},
            "upper": {"x": landmark_position[0] + 1.0, "y": landmark_position[2] + 1.0}
        }
    
    camera_positions = np.array(camera_positions)
    distances = np.linalg.norm(camera_positions - landmark_position, axis=1)
    max_distance = np.max(distances) if len(distances) > 0 else 1.0
    
    landmark_x, landmark_z = landmark_position[0], landmark_position[2]
    camera_x_coords = camera_positions[:, 0]
    camera_z_coords = camera_positions[:, 2]
    
    extent = max_distance * max_distance_factor
    
    min_x = min(np.min(camera_x_coords), landmark_x) - extent * 0.5
    max_x = max(np.max(camera_x_coords), landmark_x) + extent * 0.5
    min_z = min(np.min(camera_z_coords), landmark_z) - extent * 0.5
    max_z = max(np.max(camera_z_coords), landmark_z) + extent * 0.5
    
    return {
        "lower": {"x": min_x, "y": min_z},
        "upper": {"x": max_x, "y": max_z}
    }

def export_aligned_map_json(poses_dir, database_path, output_file):
    """Export landmarks in map.json format from aligned model with real SIFT descriptors"""
    try:
        # Read reconstruction files from aligned model
        images_file = poses_dir / "images.txt"
        points3d_file = poses_dir / "points3D.txt"
        
        if not images_file.exists() or not points3d_file.exists():
            print("Could not find reconstruction files for map export")
            return False
        
        print("Exporting aligned map.json with real SIFT descriptors...")
        
        # Read poses from aligned model (already in correct coordinate system)
        colmap_poses = read_colmap_poses(images_file)
        
        # Create mapping from COLMAP image_id to image_name and positions
        colmap_images = {}
        for image_name, pose_data in colmap_poses.items():
            colmap_images[pose_data['image_id']] = {
                'name': image_name,
                'position': pose_data['position']
            }
        
        # Parse 3D points and create landmarks (already in correct scale/coordinates)
        landmarks = []
        processed = 0
        
        with open(points3d_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    try:
                        point3d_id = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # Parse the track
                        track = []
                        observing_cameras = []
                        
                        for i in range(8, len(parts), 2):
                            if i + 1 < len(parts):
                                img_id = int(parts[i])
                                point2d_idx = int(parts[i + 1])
                                track.append((img_id, point2d_idx))
                                
                                if img_id in colmap_images:
                                    cam_pos = colmap_images[img_id]['position']
                                    observing_cameras.append(cam_pos)
                        
                        sightings = len(track)
                        
                        # Get descriptor
                        descriptor_b64 = get_descriptor_for_3d_point(
                            track, colmap_images, database_path
                        )
                        if descriptor_b64 is None:
                            continue
                        
                        # Calculate bounds with aligned positions
                        bounds = calculate_spatial_bounds(
                            landmark_position=np.array([x, y, z]),
                            camera_positions=observing_cameras,
                            max_distance_factor=2.0
                        )
                        
                        landmark = {
                            "bounds": bounds,
                            "desc": descriptor_b64,
                            "id": point3d_id,
                            "orientation": [0.0, 0.0, 1.0],
                            "position": [x, y, z],
                            "sightings": sightings
                        }
                        
                        landmarks.append(landmark)
                        processed += 1
                        
                    except (ValueError, IndexError) as e:
                        continue
        
        # Create the map structure
        map_data = {
            "landmarks": landmarks,
            "metadata": {
                "aligned_with_arkit": True,
                "total_landmarks": len(landmarks),
                "coordinate_system": "arkit_aligned"
            }
        }
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"Exported {len(landmarks)} aligned landmarks to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error exporting aligned map.json: {e}")
        return False
