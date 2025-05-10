#!/bin/bash

# Enhanced Real-time Emotion Recognition with Audio-Video Synchronization
# This script launches the improved emotion recognition application with robust audio device handling

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Enhanced Real-time Emotion Recognition with Synchronized Audio-Video${NC}\n"

# Find Python
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

# Find model
MODEL_PATH="models/dynamic_padding_no_leakage/model_best.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Emotion recognition model not found at: ${MODEL_PATH}${NC}"
    echo -e "${YELLOW}Please download the model or check the path.${NC}"
    exit 1
fi

# Find OpenSMILE
OPENSMILE_PATH=""
if [ -d "./opensmile-3.0.2-macos-armv8" ]; then
    OPENSMILE_PATH="./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/bin/SMILExtract"
fi

if [ ! -f "$OPENSMILE_PATH" ]; then
    echo -e "${YELLOW}OpenSMILE not found at default location. Searching alternatives...${NC}"
    
    # Try to find OpenSMILE in the system
    if command -v SMILExtract &>/dev/null; then
        OPENSMILE_PATH=$(which SMILExtract)
        echo -e "${GREEN}Found OpenSMILE in system path: $OPENSMILE_PATH${NC}"
    fi
    
    if [ -z "$OPENSMILE_PATH" ]; then
        echo -e "${YELLOW}OpenSMILE not found. Audio features will be limited.${NC}"
        echo -e "${YELLOW}You may need to manually specify the path to OpenSMILE.${NC}"
    fi
fi

# Find OpenSMILE config
CONFIG_PATH=""
if [ -f "./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf" ]; then
    CONFIG_PATH="./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
fi

if [ -z "$CONFIG_PATH" ] && [ ! -z "$OPENSMILE_PATH" ]; then
    # Try to find the config relative to the OpenSMILE executable
    OPENSMILE_DIR=$(dirname "$OPENSMILE_PATH")
    POTENTIAL_CONFIG="$OPENSMILE_DIR/../config/egemaps/v02/eGeMAPSv02.conf"
    if [ -f "$POTENTIAL_CONFIG" ]; then
        CONFIG_PATH="$POTENTIAL_CONFIG"
        echo -e "${GREEN}Found OpenSMILE config: $CONFIG_PATH${NC}"
    fi
fi

# Set parameters
CAMERA_INDEX=0
AUDIO_DEVICE=""  # Empty for auto-detection
WINDOW_SIZE=5    # Frames for prediction smoothing
FEATURE_WINDOW=3 # Window size in seconds for both audio and video features

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts

echo -e "\nRunning with the following configuration:"
echo -e "  - Camera: ${GREEN}Device $CAMERA_INDEX${NC}"
echo -e "  - Microphone: ${GREEN}Auto-detect${NC}"
echo -e "  - Feature window: ${GREEN}$FEATURE_WINDOW seconds${NC} (synchronized for both audio & video)"
echo -e "  - Smoothing window: ${GREEN}$WINDOW_SIZE frames${NC}"
echo -e "  - Model: ${GREEN}$MODEL_PATH${NC}"
echo -e "  - OpenSMILE: ${GREEN}$OPENSMILE_PATH${NC}"
echo -e "  - Config: ${GREEN}$CONFIG_PATH${NC}"

# Run the application
echo -e "\nStarting real-time emotion recognition..."
echo -e "Press ${YELLOW}q${NC} or ${YELLOW}ESC${NC} to quit\n"

$PYTHON_PATH scripts/enhanced_compatible_realtime_emotion.py \
    --model "$MODEL_PATH" \
    --opensmile "$OPENSMILE_PATH" \
    --config "$CONFIG_PATH" \
    --camera_index "$CAMERA_INDEX" \
    --window_size "$WINDOW_SIZE" \
    --feature_window "$FEATURE_WINDOW" \
    --fps 15 \
    --display_width 1200 \
    --display_height 700

echo -e "\nEmotion recognition application closed."
echo -e "${GREEN}Done.${NC}"
