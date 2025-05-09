#!/bin/bash
# Script to run the smoothing comparison demo

# Display header
echo "Smoothing Comparison Demo"
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
change_rate=0.3

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
        --change-rate)
            change_rate="$2"
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
echo "  Change rate: $change_rate (higher = more jumpy)"

# Add the current directory to PYTHONPATH to ensure local imports work
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/scripts"
echo "Using Python at: $(which $python_cmd)"
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# Run the main application
echo "Starting smoothing comparison demo..."
echo "Press q or ESC to quit"
echo ""
echo "This demo will show both the raw random values and the smoothed values side by side"
echo "so you can clearly see the difference that the moving average makes."
echo ""

$python_cmd scripts/compare_smoothing_demo.py \
    --fps "$fps" \
    --display_width "$display_width" \
    --display_height "$display_height" \
    --window_size "$window_size" \
    --change_rate "$change_rate"

# Check if execution succeeded
if [ $? -ne 0 ]; then
    echo "An error occurred while running the demo."
    exit 1
else
    echo "Demo stopped."
fi
