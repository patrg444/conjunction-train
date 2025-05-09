# Emotion Recognition Model Architecture

This document details the branched neural network architecture used in our emotion recognition system. The model processes audio and video features separately before merging them for final emotion classification.

## Overview

The model uses a dual-stream architecture with separate processing branches for:
1. **Audio features** (89-dimensional OpenSMILE features)
2. **Video features** (512-dimensional FaceNet embeddings)

These branches are then merged and passed through fully connected layers for final classification into 6 emotion categories.

## Audio Branch

The audio branch first applies Conv1D layers for feature extraction, then uses Bidirectional LSTM for temporal processing:

```
Audio Input (89-dim) → Masking → Conv1D → BatchNorm → MaxPooling → Conv1D → BatchNorm → MaxPooling → Bidirectional LSTM → Dropout → Bidirectional LSTM → Dense → Dropout
```

**Detailed layers:**
1. **Input**: Variable-length sequence of 89-dim OpenSMILE features
2. **Masking**: Masks zero values for variable-length processing
3. **Conv1D**: 64 filters with kernel size 3, ReLU activation
4. **BatchNorm**: Normalizes activations
5. **MaxPooling**: Reduces sequence length by factor of 2
6. **Conv1D**: 128 filters with kernel size 3, ReLU activation  
7. **BatchNorm**: Normalizes activations
8. **MaxPooling**: Further reduces sequence length
9. **Bidirectional LSTM**: 128 units, returns sequences
10. **Dropout**: 0.3 rate
11. **Bidirectional LSTM**: 64 units, returns final state
12. **Dense**: 128 units, ReLU activation
13. **Dropout**: 0.4 rate

## Video Branch

The video branch directly applies Bidirectional LSTM layers to the high-dimensional FaceNet features:

```
Video Input (512-dim) → Masking → Bidirectional LSTM → Dropout → Bidirectional LSTM → Dense → Dropout
```

**Detailed layers:**
1. **Input**: Variable-length sequence of 512-dim FaceNet features
2. **Masking**: Masks zero values for variable-length processing
3. **Bidirectional LSTM**: 256 units, returns sequences
4. **Dropout**: 0.3 rate
5. **Bidirectional LSTM**: 128 units, returns final state
6. **Dense**: 256 units, ReLU activation
7. **Dropout**: 0.4 rate

## Merged Network

After separate processing, the features from both branches are concatenated and processed:

```
Concatenate[Audio Branch, Video Branch] → Dense → BatchNorm → Dropout → Dense → BatchNorm → Dropout → Dense (Output)
```

**Detailed layers:**
1. **Concatenate**: Merges audio and video features
2. **Dense**: 256 units, ReLU activation
3. **BatchNorm**: Normalizes activations
4. **Dropout**: 0.5 rate
5. **Dense**: 128 units, ReLU activation
6. **BatchNorm**: Normalizes activations
7. **Dropout**: 0.4 rate
8. **Dense (Output)**: 6 units, softmax activation (one per emotion)

## Key Differences Between Branches

| Feature | Audio Branch | Video Branch |
|---------|--------------|--------------|
| Input Dimension | 89 | 512 |
| Uses Conv1D | Yes | No |
| First LSTM Size | 128 | 256 |
| Second LSTM Size | 64 | 128 |
| Final Dense Size | 128 | 256 |

The audio branch uses Conv1D layers because the raw OpenSMILE features benefit from local pattern extraction. The video branch skips Conv1D since FaceNet features are already highly processed embeddings.

## Handling Variable-length Sequences

The model can process sequences of any length due to:
1. **Masking layers** at the start of each branch
2. **Dynamic padding** during training/inference
3. **LSTM layers** that can handle variable-length inputs

This allows the model to process video clips of different durations without requiring fixed-length input.
