# Improved Real-time Emotion Recognition

This document describes the improvements made to the real-time emotion recognition pipeline to ensure better audio-video synchronization and overall performance.

## Synchronized Audio-Video Feature Windows

One of the key improvements in this updated version is ensuring perfect synchronization between the audio and video feature windows. The original implementation had a hard-coded audio window size that might not always match the effective video window size, potentially leading to temporal misalignment in multimodal emotion predictions.

### Key Changes:

1. **Configurable Feature Window Size**: Added a new parameter `--feature_window` that controls both the audio and video feature windows simultaneously, ensuring they remain perfectly synchronized.

2. **Consistent Buffer Management**: Modified the buffer management logic to maintain exactly the same temporal window for both modalities. This ensures that:
   - Audio features cover exactly the same time period as video features
   - Both buffers maintain the same number of frames relative to their sampling rate
   - Prediction inputs remain temporally coherent

3. **Command-line Configurability**: The feature window size can now be specified from the command line, allowing for easy testing of different temporal windows without code changes.

## Prediction Smoothing Window

The smoothing window (which averages prediction probabilities over multiple frames to reduce jumpiness) is separate from the feature window used for model inputs. This provides:

1. **Independent Control**: Feature window and smoothing window can be tuned independently for optimal performance
2. **Clear Separation of Concerns**: Audio-video feature synchronization is handled separately from output smoothing

## Usage

The updated script `run_realtime_emotion_responsive.sh` includes the improved parameters:

```bash
WINDOW_SIZE=5  # Number of frames for prediction smoothing, not seconds
FEATURE_WINDOW=3  # Window size in seconds for audio and video features (must be the same)
```

To run with different settings, simply modify these parameters in the script.

## Notes on Implementation

- The feature window represents a time duration in seconds, while the smoothing window represents a number of frames
- At 15 FPS, a 3-second feature window corresponds to approximately 45 frames of video
- The audio window now matches the video window precisely in terms of time duration
- Both modalities are truncated to the same temporal extent before making predictions
- This ensures that the model only sees temporally coherent data from both modalities
