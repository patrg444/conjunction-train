#!/bin/bash
# Script to run the face detection demo with emotion smoothing

# Display header
echo "Face Detection Demo with Smoothed Emotion"
echo "=============================================="
echo ""

# Detect Anaconda environment and use it if available
if command -v conda &> /dev/null; then
    echo "Anaconda detected, using conda Python"
    # Try to use conda python directly
    python_cmd="$(conda info --base)/bin/python"
    # If that fails, fall back to system python
    if [ ! -f "$python_cmd" ]; then
        python_cmd="python3"
    fi
else
    python_cmd="python3"
fi
echo "Using Python at: $python_cmd"

# Parameters
fps=15
display_width=1200
display_height=700
window_size=30

# Handle any command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --fps)
            fps="$2"
            shift 2
            ;;
        --width)
            display_width="$2"
            shift 2
            ;;
        --height)
            display_height="$2"
            shift 2
            ;;
        --window-size)
            window_size="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if OpenCV is available
echo "Checking for OpenCV..."
if ! $python_cmd -c "import cv2" &> /dev/null; then
    echo "Error: OpenCV (cv2) is not installed."
    echo "Please install it with: pip install opencv-python"
    exit 1
else
    echo "OpenCV is available!"
fi

# Print settings
echo ""
echo "Running with settings:"
echo "  Target FPS: $fps"
echo "  Display size: ${display_width}x${display_height}"
echo "  Smoothing window size: $window_size frames"

# Add the current directory to PYTHONPATH to ensure local imports work
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/scripts"
echo "Using Python at: $(which $python_cmd)"
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# Run the main application
echo "Starting face detection demo with smoothed emotions..."
echo "Press q or ESC to quit"
echo ""
echo "This demo will detect your face and show:"
echo "- Your face with a green box around it"
echo "- Raw emotion values simulated from face position"
echo "- Smoothed emotion values using a moving average"
echo ""

$python_cmd scripts/face_detection_demo.py \
    --fps "$fps" \
    --display_width "$display_width" \
    --display_height "$display_height" \
    --window_size "$window_size"

# Check if execution succeeded
if [ $? -ne 0 ]; then
    echo "An error occurred while running the face detection demo."
    exit 1
else
    echo "Face detection demo stopped."
fi
