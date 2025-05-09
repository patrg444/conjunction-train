#!/bin/bash
# Run the demo version of the emotion recognition system
# This version shows fully functioning feature extraction with simulated predictions

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Real-time Emotion Recognition System (DEMO MODE)${NC}\n"
echo -e "${YELLOW}Note: This demo uses realistic simulations instead of actual model predictions${NC}"
echo -e "${YELLOW}but all feature extraction is working correctly.${NC}\n"

# Check for Python
PYTHON_PATH=""
if command -v python &>/dev/null; then
    PYTHON_PATH=$(which python)
elif command -v python3 &>/dev/null; then
    PYTHON_PATH=$(which python3)
elif command -v conda &>/dev/null; then
    echo "Anaconda detected, using conda Python"
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

# Check for required Python dependencies
echo "Checking for required Python dependencies..."

echo "Checking for OpenCV..."
if $PYTHON_PATH -c "import cv2" 2>/dev/null; then
    echo -e "${GREEN}OpenCV is available!${NC}"
else
    echo -e "${RED}OpenCV is not installed. Please install it with: pip install opencv-python${NC}"
    exit 1
fi

echo "Checking for PyAudio..."
if $PYTHON_PATH -c "import pyaudio" 2>/dev/null; then
    echo -e "${GREEN}PyAudio is available!${NC}"
else
    echo -e "${RED}PyAudio is not installed. Please install it with: pip install pyaudio${NC}"
    exit 1
fi

echo "Checking for pandas..."
if $PYTHON_PATH -c "import pandas" 2>/dev/null; then
    echo -e "${GREEN}pandas is available!${NC}"
else
    echo -e "${RED}pandas is not installed. Please install it with: pip install pandas${NC}"
    exit 1
fi

echo "Checking for facenet-pytorch..."
if $PYTHON_PATH -c "from facenet_pytorch import MTCNN, InceptionResnetV1" 2>/dev/null; then
    echo -e "${GREEN}facenet-pytorch is available!${NC}"
else
    echo -e "${RED}facenet-pytorch is not installed. Please install it with: pip install facenet-pytorch${NC}"
    exit 1
fi

# Set parameters
TARGET_FPS=15
DISPLAY_WIDTH=1200
DISPLAY_HEIGHT=700
WINDOW_SIZE=45
CAMERA_INDEX=0

# Enable debugging
export DEBUG=1
export VERBOSE=1
export SHOW_FEATURES=1

echo -e "\nRunning with settings:"
echo -e "  Target FPS: ${GREEN}$TARGET_FPS${NC}"
echo -e "  Display size: ${GREEN}${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}${NC}"
echo -e "  Window size: ${GREEN}$WINDOW_SIZE frames${NC} (for smoothing)"
echo -e "  Camera index: ${GREEN}$CAMERA_INDEX${NC}"
echo -e "  OpenSMILE config: ${GREEN}$CONFIG_PATH${NC}"
echo -e "  OpenSMILE path: ${GREEN}$OPENSMILE_PATH${NC}"
echo -e "  Debug mode: ${GREEN}ENABLED${NC}"
echo -e "  Feature visualization: ${GREEN}ENABLED${NC}"

# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts
echo -e "Using Python at: ${GREEN}$PYTHON_PATH${NC}"
echo -e "Current PYTHONPATH: ${GREEN}$PYTHONPATH${NC}"

# Make the script executable
chmod +x scripts/demo_emotion_recognition.py

# Run the application
echo -e "\nStarting real-time emotion recognition in DEMO mode..."
echo -e "Press ${YELLOW}q${NC} or ${YELLOW}ESC${NC} to quit"

echo -e "\nThis demonstration application:"
echo -e "- Captures webcam feed at ${GREEN}15 fps${NC}"
echo -e "- Extracts facial features using ${GREEN}FaceNet${NC} (512-dimensional)"
echo -e "- Captures audio from microphone"
echo -e "- Extracts audio features using ${GREEN}OpenSMILE${NC} (89-dimensional)"
echo -e "- ${YELLOW}Simulates${NC} emotion predictions with realistic transitions"
echo -e "- Applies smoothing to show transitions between emotions"
echo -e "- Displays feature visualizations for complete transparency"

$PYTHON_PATH scripts/demo_emotion_recognition.py \
    --opensmile "$OPENSMILE_PATH" \
    --config "$CONFIG_PATH" \
    --fps "$TARGET_FPS" \
    --display_width "$DISPLAY_WIDTH" \
    --display_height "$DISPLAY_HEIGHT" \
    --window_size "$WINDOW_SIZE" \
    --camera_index "$CAMERA_INDEX"

echo -e "Real-time emotion recognition demo application closed."
echo -e "${GREEN}Done.${NC}"
