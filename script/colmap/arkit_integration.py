import json
import numpy as np

def load_arkit_data(frames_json_path):
    """Load ARKit frames data from JSON file"""
    with open(frames_json_path, 'r') as f:
        frames = json.load(f)
    
    print(f"Loaded {len(frames)} ARKit frames")
    return frames

def read_arkit_poses(frames):
    """Extract camera positions from ARKit frames"""
    poses = {}
    
    for frame in frames:
        image_name = f"{frame['id']:08d}_image.jpeg"
        
        # ARKit extrinsics are camera-to-world transformation
        matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
        camera_position = matrix[:3, 3]
        
        poses[image_name] = {
            'position': camera_position
        }
    
    return poses

def create_reference_file_from_arkit(arkit_frames, output_file):
    """Create reference coordinates file from ARKit frames for model_aligner"""
    with open(output_file, 'w') as f:
        for frame in arkit_frames:
            image_name = f"{frame['id']:08d}_image.jpeg"
            
            # ARKit extrinsics are camera-to-world transformation
            matrix = np.array(frame['extrinsics']).reshape(4, 4, order='F')
            camera_position = matrix[:3, 3]
            
            # Write in format: image_name X Y Z
            f.write(f"{image_name} {camera_position[0]:.6f} {camera_position[1]:.6f} {camera_position[2]:.6f}\n")
    
    print(f"Created reference coordinates file with {len(arkit_frames)} camera positions")
    return True