#!/bin/bash
# Modified script to run the real-time emotion recognition system using webcam and microphone
# With fallbacks for common issues

# Display header
echo "Real-time Emotion Recognition with Webcam (Modified)"
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

# Look for OpenSMILE in various locations
opensmile_path=""
possible_opensmile_paths=(
    "opensmile-3.0.2-macos-armv8/bin/SMILExtract"
    "opensmile-3.0.0/bin/SMILExtract"
    "opensmile/bin/SMILExtract"
    "$(which SMILExtract 2>/dev/null)"
)

for path in "${possible_opensmile_paths[@]}"; do
    if [ -f "$path" ]; then
        opensmile_path="$path"
        echo "Found OpenSMILE at: $opensmile_path"
        break
    fi
done

if [ -z "$opensmile_path" ]; then
    echo "Warning: Could not find OpenSMILE executable."
    echo "Audio feature extraction may not work properly."
    echo "You may need to install OpenSMILE or specify its path."
    opensmile_path="dummy_path_for_fallback"
fi

# Look for best model file (with common alternate names)
model_path=""
possible_model_paths=(
    "models/dynamic_padding_no_leakage/model_best.h5"  # Actual name of file
    "models/dynamic_padding_no_leakage/best_model.h5"  # Name script expects
    "models/dynamic_padding_no_leakage/final_model.h5"
    "target_model/model_best.h5"
    "target_model/best_model.h5"
    "models/best_model.h5"
    "models/model_best.h5"
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
echo "  OpenSMILE path: $opensmile_path"

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
echo "- Extracts audio features using OpenSMILE (if available)"
echo "- Predicts emotions in real-time"
echo "- Applies smoothing to reduce jumpiness"
echo ""

# First, try running the face detection demo if the full system fails
$python_cmd scripts/realtime_emotion_recognition_webcam.py \
    --model "$model_path" \
    --fps "$fps" \
    --display_width "$display_width" \
    --display_height "$display_height" \
    --window_size "$window_size" \
    --cam_index "$cam_index" \
    --opensmile_config "$opensmile_config" \
    --opensmile_path "$opensmile_path"

result=$?

# If the full system failed, run the face detection demo as fallback
if [ $result -ne 0 ]; then
    echo "Full emotion recognition failed. Falling back to face detection demo..."
    echo "This simplified demo only uses the webcam (no audio processing)."
    echo ""
    echo "Press Enter to continue..."
    read

    $python_cmd scripts/face_detection_demo.py \
        --fps "$fps" \
        --display_width "$display_width" \
        --display_height "$display_height" \
        --window_size "$window_size"

    result=$?
fi

# Check if execution succeeded
if [ $result -ne 0 ]; then
    echo "An error occurred while running the application."
    exit 1
else
    echo "Real-time emotion recognition application closed."
fi

# Clean up temp files
rm -rf temp_extracted_audio/*.wav temp_extracted_audio/*.csv 2>/dev/null

echo "Done."
