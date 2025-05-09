#!/bin/bash
# Script to run the EmotionTrack demo application

echo "Starting EmotionTrack Marketing Analytics Platform..."
echo "This application uses your webcam and microphone for real-time emotion recognition."
echo "----------------------------------------------------------------------"

# Check Python dependencies
echo "Checking dependencies..."

# Check for Flask
if ! python -c "import flask" &> /dev/null; then
    echo "Installing Flask..."
    pip install flask
fi

# Check for TensorFlow
if ! python -c "import tensorflow" &> /dev/null; then
    echo "TensorFlow is required. Please ensure it's installed via:"
    echo "pip install tensorflow"
    exit 1
fi

# Check for facenet-pytorch
if ! python -c "import facenet_pytorch" &> /dev/null; then
    echo "Installing facenet-pytorch..."
    pip install facenet-pytorch
fi

# Check for additional dependencies
if ! python -c "import cv2, numpy, soundfile" &> /dev/null; then
    echo "Installing additional dependencies..."
    pip install opencv-python numpy soundfile
fi

echo "Dependencies satisfied."
echo "----------------------------------------------------------------------"

# Launch the application
echo "Launching EmotionTrack. Open your browser at http://localhost:5000"
cd demo_app
python app.py
