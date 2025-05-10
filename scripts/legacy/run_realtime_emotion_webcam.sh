#!/bin/bash
# Script to run the real-time emotion recognition system using webcam and microphone

# Display header
echo "Real-time Emotion Recognition with Webcam"
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
window_size=45
cam_index=0
opensmile_config="scripts/test_opensmile_config.conf"
opensmile_path="opensmile-3.0.2-macos-armv8/bin/SMILExtract"

# Look for best model file
model_path=""
possible_model_paths=(
    "models/dynamic_padding_no_leakage/best_model.h5"
    "target_model/best_model.h5"
    "models/best_model.h5"
)

for path in "${possible_model_paths[@]}"; do
    if [ -f "$path" ]; then
        model_path="$path"
        echo "Found model at: $model_path"
        break
    fi
done

if [ -z "$model_path" ]; then
    echo "Warning: Could not find a pre-trained model file."
    echo "The application will still run but will display a model loading error."
    echo "Please download a model using './download_best_model.sh' first."
    model_path="models/dynamic_padding_no_leakage/best_model.h5"  # Default path as fallback
fi

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
        --window-size|--window_size)
            window_size="$2"
            shift 2
            ;;
        --cam|--camera)
            cam_index="$2"
            shift 2
            ;;
        --model)
            model_path="$2"
            shift 2
            ;;
        --opensmile-config)
            opensmile_config="$2"
            shift 2
            ;;
        --opensmile-path)
            opensmile_path="$2"
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

# Check if TensorFlow is available
echo "Checking for TensorFlow..."
if ! $python_cmd -c "import tensorflow" &> /dev/null; then
    echo "Error: TensorFlow is not installed."
    echo "Please install it with: pip install tensorflow"
    exit 1
else
    echo "TensorFlow is available!"
fi

# Check if PyAudio is available
echo "Checking for PyAudio..."
if ! $python_cmd -c "import pyaudio" &> /dev/null; then
    echo "Error: PyAudio is not installed."
    echo "Please install it with: pip install pyaudio"
    exit 1
else
    echo "PyAudio is available!"
fi

# Check if pandas is available
echo "Checking for pandas..."
if ! $python_cmd -c "import pandas" &> /dev/null; then
    echo "Error: pandas is not installed."
    echo "Please install it with: pip install pandas"
    exit 1
else
    echo "pandas is available!"
fi

# Create temp directories
mkdir -p temp_extracted_audio

# Print settings
echo ""
echo "Running with settings:"
echo "  Model: $model_path"
echo "  Target FPS: $fps"
echo "  Display size: ${display_width}x${display_height}"
echo "  Window size: $window_size frames (for smoothing)"
echo "  Camera index: $cam_index"
echo "  OpenSMILE config: $opensmile_config"

# Add the current directory to PYTHONPATH to ensure local imports work
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/scripts"
echo "Using Python at: $(which $python_cmd)"
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# Run the main application
echo "Starting real-time emotion recognition..."
echo "Press q or ESC to quit"
echo ""
echo "This application:"
echo "- Captures webcam feed at $fps fps"
echo "- Extracts facial features using FaceNet"
echo "- Captures audio from microphone"
echo "- Extracts audio features using OpenSMILE"
echo "- Predicts emotions in real-time"
echo "- Applies smoothing to reduce jumpiness"
echo ""

# Run the application
$python_cmd scripts/realtime_emotion_recognition_webcam.py \
    --model "$model_path" \
    --fps "$fps" \
    --display_width "$display_width" \
    --display_height "$display_height" \
    --window_size "$window_size" \
    --cam_index "$cam_index" \
    --opensmile_config "$opensmile_config" \
    --opensmile_path "$opensmile_path"

# Check if execution succeeded
if [ $? -ne 0 ]; then
    echo "An error occurred while running the application."
    exit 1
else
    echo "Real-time emotion recognition application closed."
fi

# Clean up temp files
rm -rf temp_extracted_audio/*.wav temp_extracted_audio/*.csv 2>/dev/null

echo "Done."
