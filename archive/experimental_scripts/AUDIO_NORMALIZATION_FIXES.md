# Audio Feature Normalization Fixes

This document explains the improvements made to the real-time emotion recognition system to ensure audio features are properly normalized, matching the exact normalization applied during training.

## Problem Overview

During the model training in `train_branched_no_leakage.py`, audio features were normalized using specific statistics:

```python
# Calculate normalization statistics on training data only
train_audio, audio_mean, audio_std = normalize_features(train_audio)
# Apply the same statistics to validation data
val_audio, _, _ = normalize_features(val_audio, mean=audio_mean, std=audio_std)
```

However, in the real-time inference code, we weren't applying the same normalization, which could cause a distribution shift and lead to incorrect predictions.

## Implementation Details

### 1. Audio Feature Normalizer Module

Created a new module (`audio_feature_normalizer.py`) that:

- Provides normalization functions matching those used during training
- Can load pre-computed statistics or use reasonable estimates
- Creates properly normalized dummy audio features for video-only mode

### 2. Extract Normalization Statistics Script

Developed a script (`extract_audio_normalization_stats.py`) to:

- Extract the mean and standard deviation from the same datasets used in training
- Compute and save these statistics for use during inference
- Support both RAVDESS and CREMA-D datasets

### 3. Integration with Audio Processing

Updated the audio processing pipeline to:

- Apply the same normalization to extracted OpenSMILE features
- Ensure the normalization is consistent with what was used in training
- Handle cases where the feature dimensions don't match

### 4. Video-Only Mode Improvements

Enhanced the video-only mode to:

- Create properly normalized dummy audio features
- Match the distribution expected by the model
- Provide better compatibility when using just the video modality

## Technical Details

### Normalization Implementation

The normalization matches the original training code:

```python
# Avoid division by zero
safe_std = np.where(std == 0, 1.0, std)
    
# Apply normalization (exactly as in training)
normalized = (features - mean) / safe_std
```

### Statistics Storage

- Statistics are stored in `models/dynamic_padding_no_leakage/audio_normalization_stats.pkl`
- The file contains both mean and standard deviation for each feature dimension
- If statistics are not available, reasonable estimates are used

## Results

The improved normalization ensures:

1. The audio features have the same distribution during inference as they did during training
2. The model receives inputs in the same feature space it was trained on
3. Predictions are more accurate and reliable for audio-based emotions

## Usage

The normalization is automatically applied in all scripts. For manual usage:

```python
import audio_feature_normalizer

# Normalize audio features
normalized_features = audio_feature_normalizer.normalize_features(raw_features)

# Create dummy normalized features for video-only mode
dummy_features = audio_feature_normalizer.create_dummy_normalized_features(seq_length)
```

## Verification

To verify the normalization statistics were computed correctly:

```bash
python scripts/extract_audio_normalization_stats.py
```

This will process the training datasets and show statistics including:
- Feature dimensions
- Number of files and frames processed
- Range of mean and standard deviation values
