# Analysis of RAVDESS and CREMA-D Feature Files

This document summarizes the analysis of the NPZ files containing extracted features from the RAVDESS and CREMA-D datasets, which are used for emotion recognition.

## File Structure

Both datasets store their features in NPZ files with a consistent structure:

- **Keys in each file**:
  - `video_features`: Video frame features extracted from VGG-Face model
  - `audio_features`: Audio features extracted using openSMILE
  - `video_timestamps`: Timestamps for each video frame
  - `audio_timestamps`: Timestamps for each audio frame
  - `emotion_label`: Integer label representing the emotion
  - `params`: Dictionary with metadata (model name, processing parameters)

## Video Features

| Dataset | Example Shape | Feature Dimension | Non-zero Elements | Value Range |
|---------|---------------|-------------------|-------------------|-------------|
| RAVDESS | (57, 4096)    | 4096              | 12.5%             | 0.0 to 0.18 |
| CREMA-D | (35, 4096)    | 4096              | 9.4%              | 0.0 to 0.21 |

Key observations:
- Both datasets use the same feature dimensionality (4096), derived from the VGG-Face model
- Features are very sparse, with ~87.5-90.6% of values being zero
- Sequence lengths vary between and within datasets
- RAVDESS typically has more frames per sample (dataset average: 57.5)
- CREMA-D typically has fewer frames per sample (dataset average: 40.0)

## Audio Features

| Dataset | Example Shape | Feature Dimension | Non-zero Elements | Value Range |
|---------|---------------|-------------------|-------------------|-------------|
| RAVDESS | (380, 89)     | 89                | 98.9%             | -156 to 2532 |
| CREMA-D | (466, 89)     | 89                | 98.9%             | -92 to 2613 |

Key observations:
- Both datasets use the same audio feature dimensionality (89)
- Audio features are dense, with only ~1.1% zeros
- Much wider value range compared to video features
- RAVDESS audio frames (dataset average: 383.8)
- CREMA-D audio frames (dataset average: 266.6)

## Timestamps and Duration

| Dataset | Example Video Duration | Example Audio Duration |
|---------|------------------------|------------------------|
| RAVDESS | 3.73 seconds           | 3.79 seconds           |
| CREMA-D | 2.27 seconds           | 2.32 seconds           |

The timestamps indicate that:
- Audio and video are approximately aligned within each sample
- RAVDESS samples are generally longer than CREMA-D samples
- Video frame rate is approximately 15-16 fps
- Audio frame rate is much higher (approximately 100-120 fps)

## Emotion Labels

Both datasets use the same emotion classification scheme in the processed files:
- 0: Neutral/Calm
- 1: Happy
- 2: Sad
- 3: Angry
- 4: Fearful
- 5: Disgust

The RAVDESS dataset originally also included a 7th emotion (Surprised) which is excluded in the 6-class model.

## Feature Distributions

The feature values show distinct patterns:

- **Video features**:
  - Extremely sparse (most values are exactly zero)
  - When non-zero, values typically range from 0.01 to 0.20
  - Distribution is heavily right-skewed

- **Audio features**:
  - Dense representation with very few zeros
  - Wide range of values with both positive and negative components
  - Most values are close to zero, but with significant outliers
  - 5th percentile: approximately -14
  - 95th percentile: approximately 700

## Implications for Model Design

These findings inform several aspects of the model architecture and training procedure:

1. **Variable sequence handling**: The model must accommodate variable length sequences, which justifies using masking layers.

2. **Feature scaling considerations**: 
   - Video features are already in a normalized range (0 to ~0.2)
   - Audio features have a much wider range and may benefit from normalization

3. **Regularization strategies**:
   - Video features being sparse might benefit from different regularization compared to audio features
   - The increased dropout rates in the improved model are justified by the risk of overfitting to sparse patterns

4. **Input shapes**:
   - Video input shape: (max_video_length, 4096)
   - Audio input shape: (max_audio_length, 89)
   - Maximum lengths should accommodate the longest sequences in the dataset

5. **Padding strategy**:
   - Zero padding is appropriate given the natural sparsity of video features
   - For audio features, zero padding is distinct from actual values (which are rarely zero)

These insights support the architectural choices in the improved branched LSTM model, particularly the use of masking, attention mechanisms, and regularization techniques.
