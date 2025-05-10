#!/bin/bash

# Enhanced Real-time Emotion Recognition with TensorFlow 2.x Compatible Model
# This script launches the enhanced compatible emotion recognition application that correctly
# handles both audio and video input and ensures proper feature dimensional ordering

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Enhanced Real-time Emotion Recognition with Fixed Dimensional Order${NC}\n"

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

# Find OpenSMILE path
OPENSMILE_PATH=""
if [ -d "./opensmile-3.0.2-macos-armv8" ]; then
    OPENSMILE_PATH="./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"
elif [ -d "./opensmile-3.0.2-linux-x64" ]; then
    OPENSMILE_PATH="./opensmile-3.0.2-linux-x64/opensmile-3.0.2-linux-x64/bin/SMILExtract"
elif [ -d "/opt/opensmile" ]; then
    OPENSMILE_PATH="/opt/opensmile/bin/SMILExtract"
elif command -v SMILExtract &>/dev/null; then
    OPENSMILE_PATH=$(which SMILExtract)
fi

if [ -z "$OPENSMILE_PATH" ] || [ ! -f "$OPENSMILE_PATH" ]; then
    echo -e "${YELLOW}OpenSMILE executable not found. Audio features will not be extracted.${NC}"
else
    echo -e "Found OpenSMILE at: ${GREEN}$OPENSMILE_PATH${NC}"
fi

# Find OpenSMILE config
CONFIG_PATH=""
if [ -f "./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf" ]; then
    CONFIG_PATH="./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
elif [ -f "./opensmile-3.0.2-linux-x64/opensmile-3.0.2-linux-x64/config/egemaps/v02/eGeMAPSv02.conf" ]; then
    CONFIG_PATH="./opensmile-3.0.2-linux-x64/opensmile-3.0.2-linux-x64/config/egemaps/v02/eGeMAPSv02.conf"
elif [ -f "/opt/opensmile/config/egemaps/v02/eGeMAPSv02.conf" ]; then
    CONFIG_PATH="/opt/opensmile/config/egemaps/v02/eGeMAPSv02.conf"
fi

if [ -z "$CONFIG_PATH" ] || [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${YELLOW}OpenSMILE config file not found. Audio features will not be extracted.${NC}"
else
    echo -e "Found OpenSMILE eGeMAPSv02 config at: ${GREEN}$CONFIG_PATH${NC}"
fi

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

echo "Checking for PyAudio..."
if $PYTHON_PATH -c "import pyaudio" 2>/dev/null; then
    echo -e "${GREEN}PyAudio is available!${NC}"
else
    echo -e "${RED}PyAudio is not installed. Please install it with: pip install pyaudio${NC}"
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
CAMERA_INDEX=0

echo -e "\nRunning with settings:"
echo -e "  Model: ${GREEN}$MODEL_PATH${NC}"
echo -e "  Target FPS: ${GREEN}$TARGET_FPS${NC}"
echo -e "  Display size: ${GREEN}${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}${NC}"
echo -e "  Window size: ${GREEN}$WINDOW_SIZE frames${NC} (for smoothing)"
echo -e "  Camera index: ${GREEN}$CAMERA_INDEX${NC}"
echo -e "  OpenSMILE config: ${GREEN}$CONFIG_PATH${NC}"
echo -e "  OpenSMILE path: ${GREEN}$OPENSMILE_PATH${NC}"

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts
echo -e "Using Python at: ${GREEN}$PYTHON_PATH${NC}"
echo -e "Current PYTHONPATH: ${GREEN}$PYTHONPATH${NC}"

# Run the enhanced application
echo -e "\nStarting enhanced real-time emotion recognition with proper dimensional ordering..."
echo -e "Press ${YELLOW}q${NC} or ${YELLOW}ESC${NC} to quit"

echo -e "\nThis enhanced application:"
echo -e "- Captures webcam feed at ${GREEN}15 fps${NC}"
echo -e "- Extracts facial features using ${GREEN}FaceNet${NC} (512 dimensions)"
echo -e "- Captures audio from microphone"
echo -e "- Extracts audio features using ${GREEN}OpenSMILE${NC} (88 dimensions)"
echo -e "- Uses EXACTLY the same feature types/dimensions as the training data"
echo -e "- Ensures ${GREEN}correct dimensional order (video first, audio second)${NC} matching the training script"
echo -e "- Automatically finds the best microphone device if none specified"
echo -e "- Shows enhanced visualization and statistics"

$PYTHON_PATH scripts/enhanced_compatible_realtime_emotion.py \
    --model "$MODEL_PATH" \
    --opensmile "$OPENSMILE_PATH" \
    --config "$CONFIG_PATH" \
    --fps "$TARGET_FPS" \
    --display_width "$DISPLAY_WIDTH" \
    --display_height "$DISPLAY_HEIGHT" \
    --window_size "$WINDOW_SIZE" \
    --camera_index "$CAMERA_INDEX"

echo -e "Enhanced real-time emotion recognition application closed."
echo -e "${GREEN}Done.${NC}"
