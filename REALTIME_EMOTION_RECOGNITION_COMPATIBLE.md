# TensorFlow 2.x Compatible Emotion Recognition

This document explains how the compatible version of the real-time emotion recognition model works with modern TensorFlow versions.

## Overview

The compatible version addresses the `time_major` parameter issue in the original model that causes errors with newer TensorFlow versions. This implementation:

- Recreates the branched model architecture from `train_branched_no_leakage.py`
- Omits the problematic `time_major` parameter from LSTM layers
- Loads the same pre-trained weights from the original model
- Processes inputs in the same way the original pipeline does
- Provides the same 6 emotion classes (anger, disgust, fear, happiness, sadness, neutral)

## The Problem

The original model, trained with an older version of TensorFlow (1.x), used the `time_major` parameter in LSTM layers. This parameter was removed or changed in TensorFlow 2.x, causing errors when loading the model directly with newer versions:

```
Unrecognized keyword arguments passed to LSTM: {'time_major': False}
```

## The Solution

Our compatible implementation addresses this issue through several approaches:

1. **Model Recreation**: We recreate the exact same architecture as the original model but without the `time_major` parameter in LSTM layers, which makes it compatible with TensorFlow 2.x.

2. **Weight Loading**: We load the pre-trained weights from the original model directly into our compatible architecture. The weights themselves are identical - only the layer construction is modified.

3. **Input/Output Compatibility**: The compatible model accepts the same inputs (FaceNet video features and OpenSMILE audio features) and produces the same output format (6 emotion class probabilities) as the original model.

## Key Components

The compatible version consists of several key files:

1. **`scripts/tensorflow_compatible_model.py`**: Contains the `EmotionRecognitionModel` class that recreates the architecture from `train_branched_no_leakage.py` without the `time_major` parameter.

2. **`scripts/compatible_realtime_emotion.py`**: A real-time application that uses the compatible model to process webcam video and microphone audio, similar to the original pipeline.

3. **`run_realtime_emotion_compatible.sh`**: A shell script to easily run the compatible real-time emotion recognition system.

## Integration with Original Pipeline

The compatible version integrates directly with the existing pipeline:

- **Same Feature Extraction**: Uses FaceNet for facial feature extraction and OpenSMILE for audio feature extraction.
- **Same Input Processing**: Handles dynamic sequence lengths through masking instead of fixed-length padding.
- **Same Pre-trained Weights**: Uses the weights from `models/dynamic_padding_no_leakage/model_best.h5`.
- **Same Output Format**: Generates the same 6 emotion class probabilities.

## Model Architecture

The architecture follows the original "branched network with dynamic padding and no data leakage" design:

```
                     ┌──────────────────┐
                     │  Audio Features  │
                     └──────────────────┘
                              │
                     ┌──────────────────┐
                     │    Masking       │
                     └──────────────────┘
                              │
                     ┌──────────────────┐
                     │ Conv1D + BatchNorm│
                     └──────────────────┘
                              │
                     ┌──────────────────┐
                     │ Conv1D + BatchNorm│
                     └──────────────────┘
                              │
                     ┌──────────────────┐
                     │Bidirectional LSTM │
                     └──────────────────┘
                              │
                              ▼
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│Video Features │──────▶    Fusion Layer  ◀──────│ Audio Branch │
└──────────────┘      └──────────────────┘      └──────────────┘
       │                       │
┌──────────────┐              │
│   Masking    │              │
└──────────────┘              │
       │                       │
┌──────────────┐              │
│Bidirect. LSTM│              │
└──────────────┘              │
       │                       │
       └───────────────────────┘
                 │
         ┌───────────────┐
         │ Dense Layers  │
         └───────────────┘
                 │
         ┌───────────────┐
         │   Softmax     │
         └───────────────┘
                 │
         ┌───────────────┐
         │   Emotions    │
         └───────────────┘
```

## Usage

To use the compatible version:

1. **Run the compatible model with the provided script**:
   ```bash
   ./run_realtime_emotion_compatible.sh
   ```

   This will:
   - Check dependencies
   - Configure paths
   - Start the real-time emotion recognition application
   - Analyze webcam video and microphone audio
   - Display emotion predictions in real-time

2. **Press `q` or `ESC` to exit**.

## Benefits Over the Original

The compatible version has several benefits:

1. **Future Compatibility**: Works with current and future TensorFlow 2.x versions.
2. **No Legacy Environment**: No need to set up a separate environment with older TensorFlow versions.
3. **Same Performance**: Provides the same emotion recognition capability as the original model.
4. **Clean Architecture**: The model code is organized in a more modular and maintainable way.

## Requirements

- TensorFlow 2.x
- OpenCV
- PyAudio
- PyTorch (for FaceNet)
- facenet-pytorch
- OpenSMILE (optional, for audio features)

## Conclusion

The compatible version provides a seamless transition to use the same emotion recognition capabilities with modern TensorFlow versions. This approach preserves the original model's performance while ensuring compatibility with current and future software environments.
