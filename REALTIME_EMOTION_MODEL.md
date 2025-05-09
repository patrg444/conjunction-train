# Real-time Emotion Recognition System

This document describes the real-time emotion recognition system that uses a webcam and microphone to detect emotions in real-time using the same feature extraction methods and model architecture that were used to process the CREMA-D and RAVDESS datasets.

## Overview

The real-time emotion recognition system combines:

1. **Video Stream Processing**: 
   - Captures webcam video at 15fps (same as training data)
   - Extracts facial features using FaceNet (same as training pipeline)
   - Maintains a sliding window of facial features

2. **Audio Stream Processing**:
   - Captures microphone audio in real-time
   - Processes audio chunks using OpenSMILE with the same configuration
   - Extracts the same audio features used in training

3. **Emotion Prediction**:
   - Uses the pre-trained branched model (no leakage, dynamic padding)
   - Combines video and audio features for multimodal prediction
   - Applies temporal smoothing to reduce jumpiness in predictions

4. **Real-time Visualization**:
   - Shows webcam feed with face detection
   - Displays raw and smoothed emotion probabilities
   - Highlights the dominant emotion

## System Requirements

- Python 3.6+
- TensorFlow 2.x
- OpenCV (cv2)
- PyAudio
- Pandas
- OpenSMILE (for audio feature extraction)
- FaceNet implementation (included)
- A trained emotion recognition model

## How It Works

### 1. Data Capture and Feature Extraction

- **Video Processing**:
  - Webcam frames are captured at exactly 15fps to match training data
  - Face detection is performed using MTCNN (via FaceNet)
  - 128-dimensional facial embeddings are extracted using FaceNet
  - A sliding window of features is maintained for temporal context

- **Audio Processing**:
  - Raw audio is captured from the microphone
  - Chunks of audio are processed using OpenSMILE
  - The same eGeMAPS feature set used in training is extracted
  - Features are continuously updated as new audio comes in

### 2. Model Prediction

- The system loads the same branched model architecture that was trained on RAVDESS and CREMA-D
- Both video and audio features are fed into the model
- The model predicts probabilities for each emotion class
- Temporal smoothing is applied to reduce noise and jumpiness

### 3. Visualization

- The webcam feed shows the detected face with a bounding box
- Raw emotion probabilities are displayed as bars
- Smoothed probabilities (using a moving average) are shown for comparison
- The dominant emotion is highlighted with its confidence score

## Running the System

Two scripts have been provided:

1. **Face Detection Demo**: 
   ```bash
   ./run_face_detection_demo.sh
   ```
   This demonstrates face detection with simulated emotion values (without requiring a trained model).

2. **Full Emotion Recognition**:
   ```bash
   ./run_realtime_emotion_webcam.sh
   ```
   This runs the complete system with video and audio processing, and requires a trained model.

### Command-line Options

Both scripts accept parameters to customize behavior:

- `--fps`: Target video capture frame rate (default: 15fps)
- `--width`, `--height`: Display window size
- `--window-size`: Temporal smoothing window size (in frames)
- `--model`: Path to the trained model file
- `--cam`: Camera index to use (default: 0)

Example:
```bash
./run_realtime_emotion_webcam.sh --window-size 30 --cam 1
```

## Implementation Details

The system is implemented as a multi-threaded application:

1. **Video Capture Thread**: Captures frames at exactly 15fps
2. **Video Feature Extraction Thread**: Extracts FaceNet embeddings
3. **Audio Capture Thread**: Captures audio from microphone
4. **Audio Feature Extraction Thread**: Processes audio with OpenSMILE
5. **Prediction Thread**: Combines features and runs through the model
6. **Display Thread**: Visualizes results and handles UI

These threads communicate using thread-safe queues and locks to ensure data consistency.

## Downsampling to 15fps

Since the model was trained on 15fps video, we implement precise 15fps downsampling:

1. For cameras that report >15fps:
   - We use frame skipping based on the camera's actual frame rate
   - For example, a 30fps camera would use every other frame

2. For cameras that report ≤15fps:
   - We capture all frames but control timing to match 15fps

This approach ensures consistent input to match what the model was trained on.

## Customizing the System

You can customize several aspects of the system:

- **Window size**: Adjust temporal smoothing (higher = smoother but more latency)
- **Camera**: Select different webcam (useful for systems with multiple cameras)
- **Model**: Point to a different trained model file
- **Display size**: Change the UI window dimensions

## Troubleshooting

Common issues:

1. **Model loading errors**: Ensure the model file exists and is compatible with your TensorFlow version
2. **Camera errors**: Try a different camera index if your webcam is not detected
3. **Audio errors**: Ensure your microphone is working and not in use by another application
4. **OpenSMILE errors**: Verify the OpenSMILE installation and config file path

## Technical Note on Feature Extraction Pipeline

This implementation uses exactly the same feature extraction pipeline as was used for training:

1. Video processing uses FaceNet with 128-dimensional embeddings
2. Audio processing uses OpenSMILE with the eGeMAPS feature set
3. The exact same preprocessing steps are applied (15fps video, audio feature rate)

This consistency is critical for the model to perform correctly, as neural networks are sensitive to differences in input data characteristics.
