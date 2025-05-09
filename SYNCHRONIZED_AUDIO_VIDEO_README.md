# Synchronized Audio-Video Emotion Recognition

This document describes the enhanced real-time emotion recognition system with synchronized audio-video processing.

## Overview

The original emotion recognition pipeline had issues with proper audio input device selection, which we've fixed in this improved version. The system:

1. Uses the same branched network architecture from the original `train_branched_no_leakage.py` script
2. Removes the problematic `time_major` parameter from LSTM layers for compatibility with newer TensorFlow versions
3. Properly selects and configures the microphone (MacBook Pro Microphone - device #1)
4. Synchronizes audio and video inputs for more accurate emotion recognition
5. Maintains the same 6-emotion classification (anger, disgust, fear, happiness, sadness, neutral)

## Components

### 1. TensorFlow Compatible Model

The `scripts/tensorflow_compatible_model.py` script provides a compatible version of the emotion recognition model that:
- Uses the same architecture as the original model
- Eliminates the problematic `time_major` parameter in LSTM layers
- Works with newer TensorFlow versions

### 2. Audio-Video Synchronized Pipeline

The `scripts/compatible_realtime_emotion.py` script:
- Captures webcam video at 15 FPS
- Processes audio from the specifically selected microphone device
- Extracts facial features using FaceNet
- Extracts audio features using OpenSMILE
- Ensures proper synchronization between audio and video frames
- Predicts emotions in real-time using our compatible model

### 3. Testing Tools

We created a simple audio-video synchronization tester:
- `scripts/test_microphone.py`: Tests microphone connectivity and lists available devices
- `scripts/test_audio_video_sync.py`: Tests the synchronization between audio and video streams

## Running the System

The system can be launched with either of these scripts:

1. `./run_emotion_sync.sh`: Our new simplified script that specifically targets the MacBook Pro microphone (device #1)
2. `./run_realtime_emotion_responsive.sh`: A more configurable script with the audio device specified

Both scripts:
- Load the same pre-trained model from `models/dynamic_padding_no_leakage/model_best.h5`
- Use the same configuration files for OpenSMILE
- Apply smoothing to reduce prediction jumpiness
- Provide a visual display with emotion probabilities

## Technical Details

### Audio Processing

The improved pipeline:
- Explicitly specifies audio device index (device #1 for MacBook Pro Microphone)
- Uses PyAudio to access the microphone with the right parameters
- Handles audio in a separate thread to avoid blocking the main video processing loop
- Maintains a sliding window of audio features for analysis

### Video Processing

The video pipeline:
- Captures frames from the webcam
- Uses FaceNet for face detection and feature extraction
- Maintains a synchronized window of video features that matches the audio feature window
- Provides real-time visualization of detected emotions

### Synchronization

To ensure accurate synchronization:
- Both audio and video use the same feature window duration (3 seconds by default)
- The audio sample rate (16kHz) and video frame rate (15 FPS) are carefully balanced
- Buffer sizes are monitored and displayed to ensure proper alignment
- Frame timestamps are tracked to maintain consistent timing

## Compatibility Notes

This improved version is designed to work with:
- Modern TensorFlow versions (2.x)
- Standard Python dependencies (PyAudio, OpenCV, NumPy)
- The pre-trained model from the original pipeline
- Both macOS and Linux environments (with appropriate device selection)

The audio device must be properly specified for your system. The test scripts will help identify the correct device index to use.
