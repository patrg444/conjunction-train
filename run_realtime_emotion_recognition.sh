#!/bin/bash
# Script to run the real-time emotion recognition system

# Display header
echo "Real-time Emotion Recognition"
echo "========================================"
echo ""

# Check for dependencies
echo "Checking dependencies..."
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

# Check Python dependencies - using a more flexible approach
# Try to import the packages with more accurate module names
$python_cmd -c "
import sys
try:
    import tensorflow
    print('✓ tensorflow')
except ImportError:
    print('✗ tensorflow - Missing dependency. Please install with: pip install tensorflow')
    sys.exit(1)

try:
    import cv2
    print('✓ cv2 (opencv)')
except ImportError:
    try:
        import opencv
        print('✓ opencv (alternative to cv2)')
    except ImportError:
        print('✗ cv2/opencv - Missing dependency. Please install with: pip install opencv-python')
        sys.exit(1)

try:
    import numpy
    print('✓ numpy')
except ImportError:
    print('✗ numpy - Missing dependency. Please install with: pip install numpy')
    sys.exit(1)

try:
    import opensmile
    print('✓ opensmile')
except ImportError:
    print('✗ opensmile - Missing dependency. Please install with: pip install opensmile')
    sys.exit(1)

try:
    import pyaudio
    print('✓ pyaudio')
except ImportError:
    print('✗ pyaudio - Missing dependency. Please install with: pip install pyaudio')
    sys.exit(1)

try:
    import matplotlib
    print('✓ matplotlib')
except ImportError:
    print('✗ matplotlib - Missing dependency. Please install with: pip install matplotlib')
    sys.exit(1)

try:
    import facenet_pytorch
    print('✓ facenet_pytorch')
except ImportError:
    print('✗ facenet_pytorch - Missing dependency. Please install with: pip install facenet-pytorch')
    sys.exit(1)
"

# Check exit code from dependency check
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Some dependencies are missing. Please install them before continuing."
    exit 1
fi

# Check for FaceNet extractor
# First make sure scripts directory is in the path
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/scripts"
echo "Setting PYTHONPATH=$PYTHONPATH"

if $python_cmd -c "from facenet_extractor import FaceNetExtractor; print('✓ FaceNet extractor')" 2>/dev/null; then
    # Success, already printed the check mark
    true
else
    echo "✗ FaceNet extractor - Trying to fix by ensuring facenet_extractor.py is in your PYTHONPATH."
    # Create a symbolic link if needed
    if [ -f "scripts/facenet_extractor.py" ]; then
        ln -sf "$(pwd)/scripts/facenet_extractor.py" .
        echo "Created symbolic link to facenet_extractor.py"
        if $python_cmd -c "from facenet_extractor import FaceNetExtractor; print('✓ FaceNet extractor (fixed)')" 2>/dev/null; then
            true # Success
        else
            echo "Still unable to import FaceNetExtractor. Check the file contents."
            exit 1
        fi
    else
        echo "Could not find scripts/facenet_extractor.py. Please make sure the file exists."
        exit 1
    fi
fi

# Check for best model
MODEL_PATH="models/branched_no_leakage_84_1/best_model.h5"
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Emotion recognition model"
else
    echo "✗ Emotion recognition model - Not found at $MODEL_PATH"
    # Try to find the model elsewhere
    found_models=$(find models -name "*.h5" | grep -i "model")
    if [ -n "$found_models" ]; then
        echo "Found alternative models:"
        echo "$found_models"
        # Use the first model found as a fallback
        MODEL_PATH=$(echo "$found_models" | head -n 1)
        echo "Using $MODEL_PATH as fallback model"
    else
        echo ""
        echo "Error: No suitable model file found. Please ensure a model file exists."
        exit 1
    fi
fi

echo ""
echo "All dependency checks passed"

# Parse command line arguments
window_size=45  # Default: 3 seconds at 15fps
fps=15
display_width=800
display_height=600

# Handle any command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --window-size)
            window_size="$2"
            shift 2
            ;;
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
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Print settings
echo ""
echo "Running with settings:"
echo "  Model: $MODEL_PATH"
echo "  Window size: $window_size"
echo "  Target FPS: $fps"
echo "  Display size: ${display_width}x${display_height}"

# Add the current directory to PYTHONPATH to ensure local imports work
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Using Python at: $(which $python_cmd)"
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# First, run the camera and LSTM registration test
echo "Testing camera access and model loading..."
$python_cmd scripts/run_emotion_recognition.py --model "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to initialize camera or load model. Please check permissions and try again."
    exit 1
fi

# Run the main application
echo "Starting real-time emotion recognition..."
echo "Press q or ESC to quit"
echo ""

$python_cmd scripts/realtime_emotion_recognition.py \
    --model "$MODEL_PATH" \
    --window_size "$window_size" \
    --fps "$fps" \
    --display_width "$display_width" \
    --display_height "$display_height"

# Check if execution succeeded
if [ $? -ne 0 ]; then
    echo "An error occurred while running the emotion recognition system."
    exit 1
else
    echo "Real-time emotion recognition stopped."
fi
