# Wav2Vec Emotion Recognition - Final Fix

## Overview

This repository contains the final fixed version of our Wav2Vec emotion recognition model. This solution successfully addresses the following issues:

1. **Dataset-Specific Emotion Coding**: Properly parses emotion codes from both CREMA-D and RAVDESS datasets using different parsing strategies
2. **NPZ Key Structure**: Uses the correct key 'wav2vec_features' to access audio features within NPZ files
3. **Continuous Emotion Indices**: Maps emotions to continuous indices (0-5) to avoid gaps in class labeling
4. **Compatible with TensorFlow Generators**: Removed class_weight parameter which is not supported for Python generator inputs

## Model Architecture

The model uses a bi-directional LSTM architecture to process wav2vec embeddings:

- Input: Variable-length sequences of 768-dimensional wav2vec features
- Two BiLSTM layers (128 units each)
- Two dense layers (256 and 128 units) with ReLU activation and dropout
- Output: 6-class softmax layer for emotion classification

## Emotion Mapping

Emotions from both datasets are mapped to a continuous index space:

- neutral/calm: 0
- happy: 1
- sad: 2
- angry: 3
- fear: 4
- disgust: 5
- surprise: 6 (if present)

## Training Process

The model is trained with the following settings:
- Adam optimizer with an initial learning rate of 0.001
- Warm-up period of 5 epochs, followed by learning rate reduction on plateau
- Early stopping based on validation accuracy with patience of 10 epochs
- The best weights are saved based on validation accuracy

## Results

The model achieves high accuracy across the combined datasets, with the best weights saved for inference.

## Scripts

- **fixed_v7_final_no_weights.py**: Main training script
- **monitor_final_fix_v7.sh**: Script to monitor training progress
- **download_final_model_v7.sh**: Script to download the trained model

## Usage

To deploy and train:
```
./deploy_final_fix_v2.sh
```

To monitor training:
```
./monitor_final_fix_v7.sh
```

To download the trained model:
```
./download_final_model_v7.sh
```
