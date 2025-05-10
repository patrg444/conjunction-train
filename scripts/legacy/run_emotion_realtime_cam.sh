#!/bin/bash
# Script to run the real-time emotion recognition camera system

# Display header
echo "Real-time Emotion Recognition Camera"
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

# Check if model exists
MODEL_PATH="models/branched_no_leakage_84_1/best_model.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH, searching for alternative models..."
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

# Parameters
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
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/scripts"
echo "Using Python at: $(which $python_cmd)"
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# Run the main application
echo "Starting real-time emotion recognition..."
echo "Press q or ESC to quit"
echo ""

$python_cmd scripts/realtime_emotion_camera.py \
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
