# Feature Normalization for Audio-Video Emotion Recognition

This document explains the feature normalization workflow for both audio and video features in the emotion recognition pipeline.

## Overview

The emotion recognition model requires properly normalized features for both audio (eGeMAPS) and video (FaceNet embeddings). The normalization process ensures that features have zero mean and unit variance across the training set, which helps with model convergence and performance.

We now use a unified normalization approach for both feature types through the `feature_normalizer.py` module.

## Normalization Flow

1. **During Training**:
   - Features are normalized using the mean and standard deviation computed from the training split
   - These statistics are saved for both audio and video features:
     - `models/dynamic_padding_no_leakage/audio_normalization_stats.pkl`
     - `models/dynamic_padding_no_leakage/video_normalization_stats.pkl`

2. **During Inference**:
   - Features are normalized using the saved statistics from training
   - If statistics are not available, reasonable estimates are used as fallback

## Using the Feature Normalizer

```python
from feature_normalizer import normalize_features, save_normalization_stats

# Normalize audio features
normalized_audio = normalize_features(audio_features, name="audio")

# Normalize video features
normalized_video = normalize_features(video_features, name="video")

# Save statistics (typically done during training)
save_normalization_stats(mean, std, name="audio")  # For audio
save_normalization_stats(mean, std, name="video")  # For video
```

## Feature Dimensions

- **Audio features**: 89-dimensional eGeMAPS features
- **Video features**: 512-dimensional FaceNet embeddings

## Generating Normalization Statistics

If needed, you can regenerate the normalization statistics using:

```bash
# For audio features
python scripts/extract_audio_normalization_stats.py

# For video features
python scripts/extract_video_normalization_stats.py
```

## Handling Missing Statistics

The feature normalizer includes fallback logic for when statistics files are missing:

1. First attempts to load from `models/dynamic_padding_no_leakage/{name}_normalization_stats.pkl`
2. If that fails, uses reasonable estimates based on feature type:
   - Audio: Zero mean, 0.5 standard deviation
   - Video: Zero mean, unit standard deviation

## Real-time Inference

All real-time inference scripts have been updated to use the unified normalizer:
- `compatible_realtime_emotion.py`
- `enhanced_compatible_realtime_emotion.py`
- `video_only_emotion.py`

## Dataset Balance

Analysis of the dataset balance shows:
- Most emotions: ~16.84% of samples each
- Neutral: 15.81% of samples

This slight imbalance is acceptable, but could be addressed with class weights if needed.

## Legacy Support

For backward compatibility, some scripts still maintain NumPy arrays:
- `audio_mean.npy`
- `audio_std.npy`
- `video_mean.npy`
- `video_std.npy`

However, all core code now uses the pickle-based approach.

## Laughter Detection Extension

The emotion recognition system now includes an auxiliary branch for laughter/humor detection, which adds the capability to identify laughs and humorous content alongside standard emotions.

### Laughter Detection Data Pipeline

1. **Data Collection**:
   - Multiple datasets are combined to build a robust laughter classifier:
     - **AudioSet**: Labeled laughter segments from YouTube videos
     - **LIRIS-ACCEDE**: Movie clips with humor ratings
     - **AMIGAIL**: Meeting corpus with laughter annotations
     - **TED-LIUM**: Speech corpus with laughter annotations
     - **MUStARD**: Sitcom humor and sarcasm corpus

2. **Data Processing**:
   - Each dataset has a dedicated processor to convert it to a unified format
   - The unified manifest is created at `datasets/manifests/laughter_v1.csv`
   - Both positive (laughter) and negative (non-laughter) examples are included

3. **Feature Pipeline Commands**:
   ```bash
   # Set up directories and process all datasets
   make laughter_data
   
   # Process individual datasets
   make audioset
   make liris_accede
   
   # Create unified manifest with negative examples
   make unified_manifest
   
   # Run tests
   make test
   ```

### Model Architecture Integration

The laughter detection is implemented as an auxiliary branch in the existing model:

```
Audio Features ─┐
                ├─ Shared Encoder ─┬─ Dense ─ Softmax (N emotions)
Video Features ─┘                  │
                                   └─ Dense ─ Sigmoid (laughter probability)
```

During training, the model optimizes:
- Main emotion classification loss (categorical crossentropy)
- Laughter detection loss (binary crossentropy) with weight = 0.3

The output includes both the emotion class and a laughter probability score (0-1).

### Using the Laughter Detection Feature

```python
# During inference, model.predict() returns two outputs:
emotion_probs, laugh_prob = model.predict([audio_features, video_features])

# Get most likely emotion
emotion_class = np.argmax(emotion_probs)

# Check if laughter is detected (using threshold)
is_laughing = laugh_prob > 0.5

# Combined interpretation
if is_laughing:
    print(f"Detected {emotion_labels[emotion_class]} with laughter (conf: {laugh_prob:.2f})")
else:
    print(f"Detected {emotion_labels[emotion_class]} (no laughter)")
```

### Data Acquisition Scripts

- `fetch_audioset_laughter.sh`: Downloads laughter segments from AudioSet
- `ingest_liris_accede.py`: Processes LIRIS-ACCEDE dataset humor segments
- `build_laughter_manifest.py`: Combines all datasets into a unified manifest

For detailed implementation, refer to files in the `datasets/scripts/` directory.
