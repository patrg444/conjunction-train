# Real-time Emotion Recognition System - User Guide

This guide explains how to use the real-time emotion recognition system that processes webcam video and microphone audio to recognize emotions.

## Available Launchers

We provide two launcher scripts to run the emotion recognition system:

### 1. Demo Version (`run_demo_emotion.sh`)

```bash
./run_demo_emotion.sh
```

This version demonstrates the full pipeline with simulated predictions. It:
- Uses your webcam and microphone
- Extracts facial features using FaceNet (512-dimensional embeddings)
- Extracts audio features using OpenSMILE (eGeMAPSv02 configuration)
- Simulates emotion predictions with realistic transitions
- Ideal for testing when you want to verify your hardware setup works correctly

### 2. Full Version (`run_full_emotion.sh`)

```bash
./run_full_emotion.sh
```

This version attempts to use a trained model for predictions with robust fallbacks:
- Automatically finds and loads the best available model
- Applies multiple strategies to fix TensorFlow compatibility issues
- Falls back to simulation mode if model loading fails
- Includes more detailed visualizations and diagnostics
- Recommended for actual emotion recognition use

## System Requirements

### Software

- Python 3.6 or higher
- OpenCV (`opencv-python`)
- TensorFlow 2.x
- PyAudio
- pandas
- facenet-pytorch
- OpenSMILE (binary included in repository)

### Hardware

- Webcam
- Microphone
- Recommended: 4GB+ RAM
- Optional: CUDA-compatible GPU for faster processing

## Troubleshooting

### Camera Issues

If the system fails to detect your webcam:
- It will automatically try indices 0, 1, and 2
- If needed, specify a different camera index by editing the launcher script

### Audio Issues

If the system fails to detect your microphone:
- Check that your microphone is properly connected
- The system will list all available audio input devices on startup
- The application will still work for face-only recognition

### Model Loading Issues

If the model fails to load (indicated by "SIMULATION MODE" warning):
- This could be due to TensorFlow version incompatibility
- The system will continue to work in simulation mode
- You can download a compatible model using the `download_best_model.sh` script

## How It Works

1. **Initialization**: The system starts by detecting your webcam and microphone
2. **Feature Extraction**: 
   - Video frames are processed by FaceNet to extract facial embeddings
   - Audio is processed by OpenSMILE to extract acoustic features
3. **Prediction**: The features are fed into a trained model (or simulation if unavailable)
4. **Visualization**: Results are displayed in a window showing:
   - Your webcam feed
   - Emotion probabilities as bars
   - Feature visualizations
   - System status information

## Exiting the Application

- Press 'q' or 'ESC' to quit the application
