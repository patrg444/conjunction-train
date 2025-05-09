# Enhanced Real-time Emotion Recognition

This document describes the enhanced version of the real-time emotion recognition system with robust audio handling and improved visualization.

## Overview

The enhanced real-time emotion recognition system provides several improvements over the original compatible implementation:

1. **Robust Audio Device Handling**: Automatically detects and uses the best available microphone, with graceful fallback options.
2. **Improved Error Handling**: Better error handling and recovery from common issues like microphone permission problems.
3. **Enhanced Visualization**: Shows detailed status information and visualizes emotion probabilities with a color-coded bar chart.
4. **Audio Level Monitoring**: Provides visual feedback about audio levels to help diagnose microphone issues.
5. **Dynamic Buffer Information**: Displays real-time information about audio and video frame buffers.
6. **Auto-discovery of OpenSMILE Components**: Automatically locates OpenSMILE executables and configuration files.

## Architecture

The system consists of the following components:

- **audio_device_utils.py**: Utility functions for audio device management and testing
- **enhanced_compatible_realtime_emotion.py**: Main application for real-time emotion recognition
- **run_enhanced_emotion_sync.sh**: Launcher script with auto-configuration
- **tensorflow_compatible_model.py**: TensorFlow 2.x compatible model loader

## Prerequisites

- Python 3.6+
- TensorFlow 2.x
- OpenCV
- PyAudio
- OpenSMILE (for audio feature extraction)
- torch/torchvision (for FaceNet face detection)
- facenet-pytorch

## Usage

Run the enhanced emotion recognition system using:

```bash
./run_enhanced_emotion_sync.sh
```

This script will:
1. Automatically find Python on your system
2. Locate the pre-trained emotion recognition model
3. Find OpenSMILE and its configuration files
4. Auto-detect the best microphone
5. Launch the real-time emotion recognition application

## Troubleshooting Audio Issues

If the system shows "Waiting for audio data" or you're not seeing emotion predictions:

1. Use the audio device testing script to list and test microphones:
   ```bash
   python scripts/test_audio_devices.py --list
   ```

2. Test a specific microphone device:
   ```bash
   python scripts/test_audio_devices.py --test DEVICE_INDEX
   ```

3. Monitor audio levels from a microphone:
   ```bash
   python scripts/test_audio_devices.py --monitor DEVICE_INDEX --duration 10
   ```

4. Find the best microphone automatically:
   ```bash
   python scripts/test_audio_devices.py --find-best
   ```

## User Interface

The enhanced application provides the following UI elements:

- **Top Left**: Status information (FPS, processing status, audio status)
- **Face Detection**: Green box around detected face, with emotion label
- **Bottom Left**: Buffer statistics (audio and video frame counts)
- **Bottom Right**: Color-coded bar chart showing emotion probabilities

## Customization

You can customize various aspects of the system by editing the `run_enhanced_emotion_sync.sh` script:

- Camera device index
- Window size for prediction smoothing
- Feature window size
- Display dimensions

For more advanced customization, edit the `enhanced_compatible_realtime_emotion.py` script directly.

## Implementation Details

### Audio Processing Pipeline

1. The system auto-detects and tests available microphones
2. Audio is captured in chunks and buffered
3. OpenSMILE extracts audio features (eGeMAPSv02 features)
4. Audio features are synchronized with video features
5. Both feature streams are fed into the branched model

### Video Processing Pipeline

1. FaceNet detects faces in each video frame
2. FaceNet extracts facial embeddings (512-dimensional)
3. Face features are synchronized with audio features
4. Both feature streams are fed into the branched model

### Key Improvements

- **Robust Audio Processing**: The system now handles audio device selection automatically and provides detailed feedback about audio capture status.
- **Improved Synchronization**: Better temporal alignment between audio and video features.
- **Enhanced Visualization**: More informative display with emotion probabilities and system status.
- **Error Resilience**: Better handling of missing or low-quality inputs.
- **Auto-configuration**: Smart detection of system components and configurations.

## Model Compatibility

This enhanced version uses the same branched network model as the original pipeline:
- Uses the model trained with dynamic padding and no data leakage
- Processes the same 6 emotion classes (anger, disgust, fear, happiness, sadness, neutral)
- Compatible with TensorFlow 2.x (no time_major parameter issues)
- Works with the same pre-trained weights from models/dynamic_padding_no_leakage/model_best.h5
