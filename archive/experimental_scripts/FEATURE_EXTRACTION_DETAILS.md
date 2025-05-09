# Feature Extraction Models in the Emotion Recognition System

This document details the specific feature extraction models used in our emotion recognition system for both audio and video modalities.

## Video Feature Extraction

### FaceNet Model
- **Library**: `facenet_pytorch`
- **Architecture**: InceptionResnetV1
- **Pre-trained Weights**: VGGFace2
- **Feature Dimension**: 512
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)

### Processing Pipeline
1. **Face Detection**: 
   - MTCNN detects faces in the input frame
   - The largest face is selected with `select_largest=True`
   - A bounding box is created around the detected face

2. **Face Preprocessing**:
   - Face is cropped from the frame using the bounding box
   - Resized to 160x160 pixels (required input size for InceptionResnetV1)
   - Pixel values are normalized to [-1, 1] range by:
     ```python
     face = (np.transpose(face, (2, 0, 1)) - 127.5) / 128.0
     ```

3. **Feature Extraction**:
   - The preprocessed face is passed through InceptionResnetV1
   - The output is a 512-dimensional embedding vector
   - No further processing is applied as the FaceNet features are already normalized

4. **Buffer Management**:
   - Features are stored in a buffer for a specific time window
   - The buffer is limited to `feature_window` seconds of data (typically 3 seconds)
   - When the buffer exceeds this limit, the oldest features are discarded

## Audio Feature Extraction

### OpenSMILE Configuration
- **Version**: OpenSMILE 3.0.2
- **Configuration**: eGeMAPSv02 (extended Geneva Minimalistic Acoustic Parameter Set v2)
- **Feature Dimension**: 89
- **Feature Type**: Frame-level acoustic descriptors

### Processing Pipeline
1. **Audio Capture**:
   - Audio is captured from the microphone at 16kHz sampling rate
   - Audio is processed in chunks of 1024 samples
   - Chunks are collected for `feature_window` seconds (typically 3 seconds)

2. **Preprocessing**:
   - Audio chunks are combined and written to a temporary WAV file
   - File is prepared with the correct sampling rate and bit depth

3. **Feature Extraction**:
   - OpenSMILE is called with eGeMAPSv02 configuration:
     ```
     SMILExtract -C eGeMAPSv02.conf -I [input.wav] -csvoutput -
     ```
   - Features are extracted for each frame in the audio
   - If OpenSMILE returns 88-dimensional features, padding is added to make 89 dimensions

4. **Normalization**:
   - Features are normalized using statistics computed from the training data:
     ```python
     normalized = (features - mean) / std
     ```
   - This normalization matches the exact process used during training
   - The mean and std are either loaded from saved statistics or estimated

5. **Buffer Management**:
   - Normalized features are stored in a buffer
   - The buffer is limited to the same time window as the video features
   - Features are synchronized with video for final prediction

## Feature Dimensions and Timing

- **Video Features**: 512-dimensional vectors at 15 FPS (frame rate is configurable)
- **Audio Features**: 89-dimensional vectors at frame rate determined by OpenSMILE
- **Time Window**: 3 seconds by default (configurable)
- **Synchronization**: Both modalities are buffered and processed together

## Dummy Audio Features for Video-Only Mode

In video-only mode, dummy audio features are created with the following properties:
- Same dimensions as real audio features (89)
- Similar normalized distribution with small random values
- Consistent with the expected inputs for the audio branch
- This allows the model to run with only video data while maintaining architectural integrity

The feature extraction processes carefully maintain the dimensionality and normalization expected by the dual-branch neural network architecture.
