#!/bin/bash

# Video-Only Emotion Recognition
# This script launches the video-only emotion recognition that doesn't rely on audio features
# Perfect for systems where audio/microphone setup is problematic

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Video-Only Emotion Recognition (fixed dimensional order)${NC}\n"

# Check for Python
PYTHON_PATH=""
if command -v python &>/dev/null; then
    PYTHON_PATH=$(which python)
elif command -v python3 &>/dev/null; then
    PYTHON_PATH=$(which python3)
elif command -v conda &>/dev/null; then
    echo "Anaconda detected using conda Python"
    PYTHON_PATH="$(conda info --base)/bin/python"
fi

if [ -z "$PYTHON_PATH" ]; then
    echo -e "${RED}Python not found. Please install Python 3.6+ or Anaconda.${NC}"
    exit 1
fi

echo -e "Using Python at: ${GREEN}$PYTHON_PATH${NC}"

# Find model
MODEL_PATH=""
POSSIBLE_MODELS=(
    "models/dynamic_padding_no_leakage/model_best.h5"
    "models/dynamic_padding_no_leakage/model_checkpoint.h5"
    "models/branched_no_leakage/model_best.h5"
    "models/branched_no_augmentation/model_best.h5"
    "models/dynamic_padding/model_best.h5"
    "models/branched/model_best.h5"
)

for model in "${POSSIBLE_MODELS[@]}"; do
    if [ -f "$model" ]; then
        MODEL_PATH="$model"
        break
    fi
done

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Emotion recognition model not found. Please download the model first.${NC}"
    exit 1
fi

echo -e "Found model at: ${GREEN}$MODEL_PATH${NC}"

# Check for required Python dependencies
echo "Checking for required Python dependencies..."

echo "Checking for OpenCV..."
if $PYTHON_PATH -c "import cv2" 2>/dev/null; then
    echo -e "${GREEN}OpenCV is available!${NC}"
else
    echo -e "${RED}OpenCV is not installed. Please install it with: pip install opencv-python${NC}"
    exit 1
fi

echo "Checking for TensorFlow..."
if $PYTHON_PATH -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}TensorFlow is available!${NC}"
else
    echo -e "${RED}TensorFlow is not installed. Please install it with: pip install tensorflow${NC}"
    exit 1
fi

echo "Checking for PyTorch and FaceNet..."
if $PYTHON_PATH -c "import torch; from facenet_pytorch import MTCNN, InceptionResnetV1" 2>/dev/null; then
    echo -e "${GREEN}PyTorch and FaceNet are available!${NC}"
else
    echo -e "${RED}PyTorch or FaceNet is not installed. Please install them with: pip install torch facenet-pytorch${NC}"
    exit 1
fi

# Set parameters
TARGET_FPS=15
DISPLAY_WIDTH=1200
DISPLAY_HEIGHT=700
WINDOW_SIZE=5

# Handle camera index with fallback options
CAMERA_INDEX="${CAMERA_INDEX:-0}"  # Use environment variable if set, otherwise default to 0
echo -e "NOTE: If camera fails, try setting a different camera index with:"
echo -e "CAMERA_INDEX=1 ./run_video_only_emotion.sh"
echo -e "Available indices are typically 0, 1, 2 for built-in and external cameras"

echo -e "\nRunning with settings:"
echo -e "  Model: ${GREEN}$MODEL_PATH${NC}"
echo -e "  Target FPS: ${GREEN}$TARGET_FPS${NC}"
echo -e "  Display size: ${GREEN}${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}${NC}"
echo -e "  Window size: ${GREEN}$WINDOW_SIZE frames${NC} (for smoothing)"
echo -e "  Camera index: ${GREEN}$CAMERA_INDEX${NC}"

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts
echo -e "Using Python at: ${GREEN}$PYTHON_PATH${NC}"
echo -e "Current PYTHONPATH: ${GREEN}$PYTHONPATH${NC}"

# Run the application
echo -e "\nStarting video-only emotion recognition with dummy audio (fixed dimensional order)..."
echo -e "${YELLOW}No audio/microphone needed for this version!${NC}"
echo -e "Press ${YELLOW}q${NC} or ${YELLOW}ESC${NC} to quit"

echo -e "\nThis application:"
echo -e "- Captures webcam feed at ${GREEN}15 fps${NC}"
echo -e "- Extracts facial features using ${GREEN}FaceNet${NC} (512 dimensions)"
echo -e "- Uses EXACTLY the same feature architecture as the training data"
echo -e "- Creates dummy zero audio features (89 dimensions) for compatibility"
echo -e "- Ensures ${GREEN}correct dimensional order${NC} matching the training script"
echo -e "- Shows real-time emotion probability visualization"
echo -e "- ${YELLOW}Doesn't require microphone or OpenSMILE${NC}"

$PYTHON_PATH scripts/video_only_emotion.py \
    --model "$MODEL_PATH" \
    --fps "$TARGET_FPS" \
    --display_width "$DISPLAY_WIDTH" \
    --display_height "$DISPLAY_HEIGHT" \
    --window_size "$WINDOW_SIZE" \
    --camera_index "$CAMERA_INDEX"

echo -e "Video-only emotion recognition application closed."
echo -e "${GREEN}Done.${NC}"
