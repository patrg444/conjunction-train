#!/bin/bash

# Real-time Emotion Recognition with Manual Audio Device Selection
# This script launches the emotion recognition with a specified audio device

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default device index (can be overridden)
AUDIO_DEVICE=1

# Check for command line arguments
if [ $# -gt 0 ]; then
    AUDIO_DEVICE=$1
fi

echo -e "${BLUE}Real-time Emotion Recognition with Audio Device ${AUDIO_DEVICE}${NC}\n"

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
    echo -e "${YELLOW}OpenSMILE not found. Audio features will be limited.${NC}"
fi

# Find OpenSMILE config
CONFIG_PATH=""
if [ -f "./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf" ]; then
    CONFIG_PATH="./opensmile-3.0.2-macos-armv8/opensmile-3.0.2-macos-armv8/config/egemaps/v02/eGeMAPSv02.conf"
fi

# Set parameters
CAMERA_INDEX=0
WINDOW_SIZE=5   # Frames for prediction smoothing
FEATURE_WINDOW=3  # Window size in seconds for both audio and video features

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts

# List available audio devices for reference
echo -e "\n=== Available Audio Input Devices ==="
$PYTHON_PATH -c "
import pyaudio
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    device_info = p.get_device_info_by_index(i)
    name = device_info.get('name')
    inputs = device_info.get('maxInputChannels')
    if inputs > 0:
        print(f'Device {i}: {name} ({inputs} inputs)')
p.terminate()
"

echo -e "\nRunning with the following configuration:"
echo -e "  - Camera: ${GREEN}Device $CAMERA_INDEX${NC}"
echo -e "  - Microphone: ${GREEN}Device $AUDIO_DEVICE${NC}"
echo -e "  - Feature window: ${GREEN}$FEATURE_WINDOW seconds${NC} (synchronized for both audio & video)"
echo -e "  - Smoothing window: ${GREEN}$WINDOW_SIZE frames${NC}"
echo -e "  - Model: ${GREEN}$MODEL_PATH${NC}"

# Run the application
echo -e "\nStarting real-time emotion recognition..."
echo -e "Press ${YELLOW}q${NC} or ${YELLOW}ESC${NC} to quit\n"

$PYTHON_PATH scripts/compatible_realtime_emotion.py \
    --model "$MODEL_PATH" \
    --opensmile "$OPENSMILE_PATH" \
    --config "$CONFIG_PATH" \
    --camera_index "$CAMERA_INDEX" \
    --audio_device "$AUDIO_DEVICE" \
    --fps 15 \
    --display_width 1200 \
    --display_height 700 \
    --window_size "$WINDOW_SIZE" \
    --feature_window "$FEATURE_WINDOW"

echo -e "\nEmotion recognition application closed."
echo -e "${GREEN}Done.${NC}"
