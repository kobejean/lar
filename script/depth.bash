#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 input/iimori1"
    exit 1
fi

DIRECTORY="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist."
    exit 1
fi

# Model and parameters
MODEL="script/depth/DepthAnythingV2SmallF16.mlpackage"
OUTPUT_WIDTH=512
OUTPUT_HEIGHT=384

# Check if model exists
if [ ! -d "$MODEL" ]; then
    echo "Error: Model '$MODEL' does not exist."
    exit 1
fi

# Check if DepthCLI exists
if [ ! -f "script/depth/DepthCLI" ]; then
    echo "Error: DepthCLI not found at 'script/depth/DepthCLI'"
    exit 1
fi

echo "Processing images in directory: $DIRECTORY"
echo "Model: $MODEL"
echo "Output size: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo ""

# Counter for processed files
processed=0
failed=0

# Process all image files in the directory
for image_file in "$DIRECTORY"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    # Check if file exists (handles case where no files match the pattern)
    if [ ! -f "$image_file" ]; then
        continue
    fi
    
    # Extract filename without extension
    filename=$(basename "$image_file")
    name_without_ext="${filename%.*}"
    
    # Create output filename
    output_file="${DIRECTORY}/${name_without_ext}_depth.pfm"
    
    echo "Processing: $filename"
    echo "  Input:  $image_file"
    echo "  Output: $output_file"
    
    # Run DepthCLI
    if ./script/depth/DepthCLI \
        --model "$MODEL" \
        --output-width "$OUTPUT_WIDTH" \
        --output-height "$OUTPUT_HEIGHT" \
        --input "$image_file" \
        --output "$output_file"; then
        
        echo "  ✓ Success"
        ((processed++))
    else
        echo "  ✗ Failed"
        ((failed++))
    fi
    echo ""
done

echo "Processing complete!"
echo "Successfully processed: $processed files"
echo "Failed: $failed files"

if [ $processed -eq 0 ]; then
    echo "No image files found in directory '$DIRECTORY'"
    echo "Supported formats: jpg, jpeg, png (case insensitive)"
fi