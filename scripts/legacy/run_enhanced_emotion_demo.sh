#!/bin/bash
# Script to run the enhanced real-time emotion recognition demo

# ANSI colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}     ENHANCED REAL-TIME EMOTION RECOGNITION DEMO     ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${YELLOW}This demo uses the webcam and microphone to recognize emotions in real-time.${NC}"
echo -e "${YELLOW}Using model: models/branched_no_leakage_84_1/best_model.h5${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""

# Check if model exists
if [ ! -f "models/branched_no_leakage_84_1/best_model.h5" ]; then
    echo -e "${RED}Error: Model file not found at models/branched_no_leakage_84_1/best_model.h5${NC}"
    echo -e "${YELLOW}Please ensure the model file exists before running this demo.${NC}"
    exit 1
fi

# Check for Python dependencies
echo -e "${YELLOW}Checking for required dependencies...${NC}"
python -c "
import sys
missing = []
try:
    import cv2
except ImportError:
    missing.append('opencv-python')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import tensorflow
except ImportError:
    missing.append('tensorflow')
try:
    import pyaudio
except ImportError:
    missing.append('pyaudio')
try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')
try:
    import opensmile
except ImportError:
    missing.append('opensmile')
try:
    import torch
except ImportError:
    missing.append('torch')
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    missing.append('facenet-pytorch')

if missing:
    print('Missing dependencies: ' + ', '.join(missing))
    sys.exit(1)
else:
    print('All required dependencies are installed.')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Some dependencies are missing. Please install them using:${NC}"
    echo -e "${YELLOW}pip install opencv-python numpy tensorflow pyaudio matplotlib opensmile torch facenet-pytorch${NC}"
    exit 1
fi

echo -e "${GREEN}All dependencies are installed.${NC}"

# Make the script executable
chmod +x scripts/enhanced_realtime_emotion_demo.py

# Run the demo
echo -e "${YELLOW}Starting the enhanced real-time emotion recognition demo...${NC}"
echo -e "${YELLOW}Press 'q' or 'ESC' to exit the demo.${NC}"
echo ""

python scripts/enhanced_realtime_emotion_demo.py --model models/branched_no_leakage_84_1/best_model.h5 --fps 15

echo -e "${GREEN}Demo completed.${NC}"
