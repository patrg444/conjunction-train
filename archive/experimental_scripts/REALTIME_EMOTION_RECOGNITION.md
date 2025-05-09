# Real-time Emotion Recognition System

This system provides a real-time facial and vocal emotion recognition pipeline that uses the same feature extraction process and model architecture as our trained model. It captures and analyzes both visual and audio inputs simultaneously to provide robust emotion prediction.

![Real-time Emotion Recognition Demo](https://raw.githubusercontent.com/username/repo/main/demo_screenshot.png)

## Overview

The real-time emotion recognition system:

- Captures video from your webcam at 15 FPS
- Extracts 512-dimensional facial embedding features using FaceNet
- Records audio from your microphone
- Extracts audio features using OpenSMILE with the eGeMAPSv02 configuration
- Processes both streams through our branched neural network model
- Displays real-time emotion predictions with a smoothing window
- Visualizes emotion probabilities in a user-friendly interface

## How It Works

The system uses a dual-stream architecture that processes facial and vocal features separately before combining them:

1. **Video Processing:**
   - Captures frames from the webcam at 15 FPS
   - Detects faces using MTCNN
   - Extracts facial embeddings using FaceNet's InceptionResNetV1
   - Feeds these features into the visual branch of the model

2. **Audio Processing:**
   - Records audio chunks from the microphone
   - Processes each chunk using OpenSMILE with eGeMAPSv02 configuration
   - Extracts acoustic features (pitch, intensity, spectral, etc.)
   - Feeds these features into the acoustic branch of the model

3. **Emotion Recognition:**
   - The model combines both modalities to predict emotions
   - Applies a smoothing window to reduce prediction flickering
   - Displays both the dominant emotion and probability distribution

## Running the Application

```bash
# Start the real-time emotion recognition system
./run_realtime_emotion_pipeline.sh
```

The script will automatically:
- Detect your Python environment
- Find OpenSMILE installation
- Locate the best available model
- Check for dependencies
- Configure input/output settings
- Launch the real-time recognition system

Press 'q' or ESC to exit the application.

## Key Components

### 1. `scripts/realtime_emotion_with_original_pipeline.py`

The main application that:
- Manages webcam capture and audio recording
- Processes both modalities in real-time
- Handles the UI display and visualization
- Provides compatibility with the trained model

### 2. `run_realtime_emotion_pipeline.sh`

A launcher script that:
- Sets up the environment
- Finds dependencies
- Configures the application
- Provides user feedback

### 3. `scripts/debug_realtime_emotion.py`

A diagnostic tool that tests each component separately:
- Checks camera access
- Verifies model loading
- Tests FaceNet initialization
- Validates OpenSMILE functionality
- Examines audio recording capabilities

## Troubleshooting

### Common Issues

1. **Model Loading Error:**
   - Symptoms: Application starts but shows "Will run in face detection only mode"
   - Fix: We've implemented a compatibility layer to handle different TensorFlow versions

2. **Camera Access Issues:**
   - Symptoms: "Failed to capture frame from camera"
   - Fix: The application will automatically try different camera indices (0, 1, 2)
   - Or specify a camera index manually with `--camera_index X`

3. **OpenSMILE Not Found:**
   - Symptoms: "OpenSMILE executable not found" 
   - Fix: Ensure OpenSMILE is installed or specify its path with `--opensmile /path/to/SMILExtract`

4. **Audio Problems:**
   - Symptoms: "Audio: NOT DETECTED" in the interface
   - Fix: Check your microphone permissions and settings

### Running Diagnostics

```bash
# Run diagnostic tests on all components
python scripts/debug_realtime_emotion.py
```

## Requirements

- Python 3.6+
- OpenCV
- TensorFlow 2.x
- PyAudio
- pandas
- facenet-pytorch
- OpenSMILE (for audio feature extraction)
- A webcam
- A microphone

## Advanced Options

You can customize the application with these command-line arguments:

```bash
python scripts/realtime_emotion_with_original_pipeline.py \
    --model path/to/model.h5 \
    --opensmile path/to/SMILExtract \
    --config path/to/config.conf \
    --fps 15 \
    --display_width 1200 \
    --display_height 700 \
    --window_size 45 \
    --camera_index 0
```

## Performance Notes

- Recommended: Intel i5/i7 or AMD Ryzen 5/7 or better
- RAM: 8GB+ recommended
- GPU: Not required, but improves prediction speed
- For optimal performance, ensure no other webcam applications are running
