#!/bin/bash

# COLMAP Depth Map Generation Script for Linux
# Usage: ./colmap_depth_linux.sh <image_directory>
# Example: ./colmap_depth_linux.sh input/aizu-park

set -e  # Exit on any error

# Set up library paths for Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Common library paths for different Linux distributions
    export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib64:/opt/local/lib:$LD_LIBRARY_PATH"
    
    # Check for CUDA libraries if available (for GPU acceleration)
    if [ -d "/usr/local/cuda/lib64" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        echo "CUDA libraries detected - but COLMAP may not be built with CUDA support"
    fi
    
    # Check for system-installed COLMAP dependencies
    COMMON_LIB_DIRS=("/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu")
    for lib_dir in "${COMMON_LIB_DIRS[@]}"; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        fi
    done
fi

# Check if COLMAP is installed
if ! command -v colmap &> /dev/null; then
    echo "Error: COLMAP is not installed or not in PATH"
    echo "Please install COLMAP first:"
    echo "  Ubuntu/Debian: sudo apt install colmap"
    echo "  Fedora/RHEL: sudo dnf install colmap"
    echo "  Arch: sudo pacman -S colmap"
    echo "  Or build from source: https://colmap.github.io/install.html"
    exit 1
fi

# Check if Python3 is available for depth map conversion
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required for depth map conversion"
    echo "Please install Python3: sudo apt install python3 python3-numpy"
    exit 1
fi

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_directory>"
    echo "Example: $0 input/aizu-park"
    exit 1
fi

IMAGE_DIR="$1"
BASE_DIR=$(dirname "$IMAGE_DIR")
PROJECT_NAME=$(basename "$IMAGE_DIR")

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory $IMAGE_DIR does not exist"
    exit 1
fi

# Check if images exist in the directory (excluding depth maps and other generated files)
if [ -z "$(find "$IMAGE_DIR" -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" 2>/dev/null | grep -v -E '_(depth|confidence|colmap_depth)\..*$')" ]; then
    echo "Error: No source images found in $IMAGE_DIR"
    exit 1
fi

# Create workspace directory
WORKSPACE_DIR="$IMAGE_DIR/colmap_workspace"
TEMP_IMAGES_DIR="$WORKSPACE_DIR/images"
DATABASE_PATH="$WORKSPACE_DIR/database.db"
SPARSE_DIR="$WORKSPACE_DIR/sparse"
DENSE_DIR="$WORKSPACE_DIR/dense"

echo "Setting up COLMAP workspace..."
rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$TEMP_IMAGES_DIR"
mkdir -p "$SPARSE_DIR"
mkdir -p "$DENSE_DIR"

# Copy only source images to temporary directory
echo "Copying source images to temporary directory..."

# Use the simple method that works
cp "$IMAGE_DIR"/*.jpeg "$TEMP_IMAGES_DIR/" 2>/dev/null || {
    echo "No .jpeg files found"
}

# Also try .jpg and .png if they exist
cp "$IMAGE_DIR"/*.jpg "$TEMP_IMAGES_DIR/" 2>/dev/null || true
cp "$IMAGE_DIR"/*.png "$TEMP_IMAGES_DIR/" 2>/dev/null || true

copied_count=$(ls -1 "$TEMP_IMAGES_DIR" 2>/dev/null | wc -l)
echo "Copied $copied_count source images for processing"

if [ "$copied_count" -eq 0 ]; then
    echo "ERROR: No images were copied. Exiting."
    exit 1
fi

echo "Images in workspace:"
ls -1 "$TEMP_IMAGES_DIR"

# Detect available CPU cores for parallel processing
NUM_THREADS=$(nproc 2>/dev/null || echo "4")
echo "Using $NUM_THREADS CPU threads for processing"

echo "Step 1: Feature extraction..."
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$TEMP_IMAGES_DIR" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.mask_path "" \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 4096 \
    --SiftExtraction.num_threads "$NUM_THREADS" \
    --SiftExtraction.use_gpu 0

echo "Step 2: Feature matching..."
QT_QPA_PLATFORM=offscreen colmap sequential_matcher \
    --database_path "$DATABASE_PATH" \
    --SequentialMatching.overlap 10 \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.num_threads "$NUM_THREADS" \
    --SiftMatching.use_gpu 1

echo "Step 3: Sparse reconstruction..."
# Try automatic reconstruction first, which is often more robust
QT_QPA_PLATFORM=offscreen colmap automatic_reconstructor \
    --workspace_path "$WORKSPACE_DIR" \
    --image_path "$TEMP_IMAGES_DIR" \
    --quality high \
    --camera_model PINHOLE \
    --single_camera 1 \
    --num_threads "$NUM_THREADS" || {
    
    echo "Automatic reconstruction failed, trying manual mapper..."
    QT_QPA_PLATFORM=offscreen colmap mapper \
        --database_path "$DATABASE_PATH" \
        --image_path "$TEMP_IMAGES_DIR" \
        --output_path "$SPARSE_DIR" \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.ba_refine_extra_params 0 \
        --Mapper.num_threads "$NUM_THREADS"
}

# Check if sparse reconstruction was successful
if [ ! -d "$SPARSE_DIR/0" ]; then
    echo "Error: Sparse reconstruction failed. No model generated."
    exit 1
fi

echo "Step 4: Preparing for dense reconstruction..."
# Copy the sparse model to dense directory structure that COLMAP expects
mkdir -p "$DENSE_DIR/sparse"
cp -r "$SPARSE_DIR/0"/* "$DENSE_DIR/sparse/"

# Create the images directory in dense workspace and copy images
mkdir -p "$DENSE_DIR/images"
cp "$TEMP_IMAGES_DIR"/* "$DENSE_DIR/images/"

echo "Step 5: Patch match stereo..."
# Check if GPU is available and CUDA is installed
GPU_AVAILABLE=0
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "GPU detected - using GPU acceleration"
        GPU_AVAILABLE=1
    fi
fi

if [ "$GPU_AVAILABLE" -eq 1 ]; then
    # Use GPU acceleration if available
    QT_QPA_PLATFORM=offscreen colmap patch_match_stereo \
        --workspace_path "$DENSE_DIR" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.gpu_index 0 || {
        
        echo "GPU stereo failed, falling back to CPU..."
        GPU_AVAILABLE=0
    }
fi

if [ "$GPU_AVAILABLE" -eq 0 ]; then
    # Use CPU processing
    echo "Using CPU processing (this may take longer)..."
    QT_QPA_PLATFORM=offscreen colmap patch_match_stereo \
        --workspace_path "$DENSE_DIR" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.gpu_index -1 || {
        
        echo "Patch match stereo failed. Let's try image undistorter first..."
        
        # Remove existing images to avoid conflicts
        rm -rf "$DENSE_DIR/images"
        
        # Run image undistorter to set up proper workspace
        QT_QPA_PLATFORM=offscreen colmap image_undistorter \
            --image_path "$TEMP_IMAGES_DIR" \
            --input_path "$DENSE_DIR/sparse" \
            --output_path "$DENSE_DIR" \
            --output_type COLMAP
        
        # Then try patch match stereo again
        QT_QPA_PLATFORM=offscreen colmap patch_match_stereo \
            --workspace_path "$DENSE_DIR" \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true \
            --PatchMatchStereo.gpu_index -1
    }
fi

echo "Step 6: Stereo fusion..."
QT_QPA_PLATFORM=offscreen colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply" \
    --StereoFusion.num_threads "$NUM_THREADS"

echo "Step 7: Extracting individual depth maps..."
STEREO_DIR="$DENSE_DIR/stereo"
DEPTH_MAPS_DIR="$STEREO_DIR/depth_maps"

if [ ! -d "$DEPTH_MAPS_DIR" ]; then
    echo "Error: Depth maps directory not found at $DEPTH_MAPS_DIR"
    exit 1
fi

# Create Python script for depth map conversion
PYTHON_CONVERTER=$(cat << 'EOF'
import numpy as np
import struct
import sys

def read_colmap_depth_map(path):
    try:
        with open(path, 'rb') as f:
            width = struct.unpack('<i', f.read(4))[0]
            height = struct.unpack('<i', f.read(4))[0]
            channels = struct.unpack('<i', f.read(4))[0]
            
            if channels == 1:
                depth_map = np.frombuffer(f.read(), dtype=np.float32).reshape(height, width)
            else:
                print(f'Unexpected number of channels: {channels}')
                return None
        return depth_map
    except Exception as e:
        print(f'Error reading depth map: {e}')
        return None

def write_pfm(filename, image, scale=1):
    try:
        with open(filename, 'wb') as f:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            if len(image.shape) == 3:
                color = True
                h, w, c = image.shape
                assert c == 3
            else:
                color = False
                h, w = image.shape
            
            f.write(b'PF\n' if color else b'Pf\n')
            f.write(f'{w} {h}\n'.encode())
            f.write(f'{-scale}\n'.encode())
            
            if not color:
                image = np.flipud(image)
            image.tofile(f)
        return True
    except Exception as e:
        print(f'Error writing PFM: {e}')
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 converter.py <input_depth_map> <output_pfm>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    depth_map = read_colmap_depth_map(input_file)
    if depth_map is not None:
        if write_pfm(output_file, depth_map):
            print("Successfully converted depth map")
            sys.exit(0)
    
    print("Failed to convert depth map")
    sys.exit(1)
EOF
)

# Save the Python converter to a temporary file
CONVERTER_SCRIPT="/tmp/colmap_depth_converter_$$.py"
echo "$PYTHON_CONVERTER" > "$CONVERTER_SCRIPT"

# Process each original image and find corresponding depth map (excluding generated files)
count=0
for img_file in "$IMAGE_DIR"/*.jpeg "$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.png; do
    [ -f "$img_file" ] || continue
    
    img_basename=$(basename "$img_file")
    img_name="${img_basename%.*}"
    
    # Skip files that are depth maps or other generated files
    if [[ "$img_basename" =~ _(depth|confidence|colmap_depth)\. ]]; then
        echo "Skipping generated file: $img_basename"
        continue
    fi
    
    # COLMAP generates depth maps with .geometric.bin extension
    depth_file="$DEPTH_MAPS_DIR/${img_basename}.geometric.bin"
    
    if [ -f "$depth_file" ]; then
        # Output filename in the requested format
        output_file="$IMAGE_DIR/${img_name}_colmap_depth.pfm"
        
        echo "Converting depth map for $img_basename..."
        
        # Use Python script to convert the binary depth map
        if python3 "$CONVERTER_SCRIPT" "$depth_file" "$output_file"; then
            echo "✓ Created: $output_file"
            ((count++))
        else
            echo "✗ Failed to create depth map for $img_basename"
        fi
    else
        echo "⚠ No depth map found for $img_basename"
    fi
done

# Clean up temporary converter script
rm -f "$CONVERTER_SCRIPT"

echo ""
echo "COLMAP processing complete!"
echo "Generated $count depth maps in $IMAGE_DIR"
echo "Workspace saved in: $WORKSPACE_DIR"

# Display system information
echo ""
echo "System Information:"
echo "- OS: $(uname -s) $(uname -r)"
echo "- CPU cores used: $NUM_THREADS"
echo "- GPU acceleration: $([ "$GPU_AVAILABLE" -eq 1 ] && echo "Enabled" || echo "Disabled")"
echo "- COLMAP version: $(colmap --version 2>/dev/null | head -n1 || echo "Unknown")"

# Optional: Clean up workspace to save space (uncomment if desired)
# echo "Cleaning up workspace..."
# rm -rf "$WORKSPACE_DIR"