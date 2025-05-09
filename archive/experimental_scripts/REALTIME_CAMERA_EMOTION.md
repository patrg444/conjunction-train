# Real-time Emotion Recognition with Camera

This document describes the real-time emotion recognition system that uses your computer's camera and microphone to detect emotions from facial expressions and voice in real-time.

## Overview

The real-time emotion recognition system combines:

- **Video processing**: Extracts facial features using FaceNet from your webcam at 15fps
- **Audio processing**: Captures audio from your microphone and extracts features using OpenSMILE
- **Emotion classification**: Uses a branched neural network model to predict emotions from both modalities
- **Real-time visualization**: Displays the detected emotion and confidence levels in real-time

The system is designed to work with the branched model architecture that was trained on RAVDESS and CREMA-D datasets with dynamic padding to handle variable-length inputs.

## Requirements

- Python 3.6+
- Tensorflow 2.x
- OpenCV (cv2)
- OpenSMILE
- PyAudio
- numpy
- FaceNet PyTorch
- A working webcam and microphone

## Running the System

We now have two versions of the real-time camera demo:

### 1. Basic Camera Demo

This demonstrates the camera functionality with randomly changing emotions:

```bash
./run_camera_demo.sh
```

### 2. Smooth Camera Demo

This version uses a moving average to stabilize the emotion display, making the transitions between emotions much smoother:

```bash
./run_smooth_camera_demo.sh
```

### 3. Smoothing Comparison Demo

This special demo clearly illustrates how the smoothing works by showing both raw and smoothed values side by side:

```bash
./run_compare_smoothing.sh
```

You can adjust the smoothing window size and change rate:

```bash
./run_compare_smoothing.sh --window-size 40 --change-rate 0.5
```

### 4. Face Detection Demo

This demo detects your face and shows emotion values that change based on your face position, with smoothing applied:

```bash
./run_face_detection_demo.sh
```

Features:
- Real-time face detection with OpenCV
- Simulated emotion values based on your face position and size
- Side-by-side comparison of raw vs. smoothed emotion values
- Face detection statistics

### 5. Full Real-time Emotion Recognition (Model Required)

The complete version with model integration (requires trained model file):

```bash
./run_emotion_realtime_cam.sh
```

### Command-line Options

All of the scripts accept similar command-line options to customize their behavior:

- `--model PATH`: Path to the trained model file (default: models/branched_no_leakage_84_1/best_model.h5)
- `--window-size SIZE`: Size of the feature window (default: 45 frames, which is 3 seconds at 15fps)
- `--fps FPS`: Target frames per second for video processing (default: 15)
- `--width WIDTH`: Display window width (default: 800)
- `--height HEIGHT`: Display window height (default: 600)

Example:

```bash
./run_emotion_realtime_cam.sh --window-size 30 --fps 10
```

## Usage

When the application is running:

- The main window shows the webcam feed with emotion overlay
- The current detected emotion is displayed with its confidence level
- Bar charts show probabilities for all emotions
- Press 'q' or ESC to quit the application

## How It Works

1. **Video Processing**:
   - Captures frames from your webcam at the target FPS
   - Detects faces and extracts facial features using FaceNet
   - Stores features in a sliding window buffer

2. **Audio Processing**:
   - Captures audio from your microphone in small chunks
   - Extracts acoustic features using OpenSMILE
   - Stores features in a sliding window buffer

3. **Emotion Prediction**:
   - Combines video and audio features from the sliding windows
   - Feeds them into the trained model to predict emotion
   - Updates the display with the predicted emotion and confidence levels

## Troubleshooting

If you encounter issues:

- **Camera not found**: Try specifying a different camera index by modifying the script
- **Model loading error**: Ensure you have the correct model file at the specified path
- **Dependencies missing**: Install missing dependencies with pip
- **Performance issues**: Try reducing the FPS or window size

## Implementation Details

The implementation uses a custom model loading approach to ensure compatibility with different versions of TensorFlow. It directly modifies the model configuration to remove incompatible parameters before loading.

Audio and video processing run in separate threads to optimize performance, with synchronization for emotion prediction.

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Video Capture  │    │  Audio Capture  │    │ Feature Buffers  │
│   (OpenCV)      │───▶│   (PyAudio)     │───▶│                  │
└─────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ FaceNet Feature │    │ OpenSMILE Audio │    │ Emotion Model    │
│   Extraction    │───▶│  Feature Extr.  │───▶│  (TensorFlow)    │
└─────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                      │
                                                      ▼
                                              ┌──────────────────┐
                                              │  Real-time UI    │
                                              │   Display        │
                                              └──────────────────┘
