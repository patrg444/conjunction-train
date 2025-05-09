# ATTN-CRNN v2 Model Improvements

This document outlines the enhancements made to the ATTN-CRNN (Attention-based Convolutional Recurrent Neural Network) model for emotion recognition using WAV2VEC features.

## Key Improvements

The v2 model includes several significant improvements over the original architecture:

### 1. Architecture Enhancements

- **Improved Attention Mechanism**: Enhanced self-attention with LayerNormalization and scaled attention to better capture the relationships between different time steps in the audio signals.
- **Expanded Dense Layer**: Increased the dense layer capacity from 128 to 256 units with BatchNormalization to allow for better representation learning.
- **GELU Activation**: Replaced ReLU with GELU (Gaussian Error Linear Unit) activation, which often performs better for audio tasks.
- **Increased Dropout**: More aggressive dropout (0.3 for both regular and recurrent dropout) to reduce overfitting, particularly beneficial for the complex WAV2VEC features.

### 2. Training Enhancements

- **Data Augmentation**:
  - Additive Gaussian noise (σ = 0.005) to the WAV2VEC features to improve robustness.
  - MixUp augmentation (α = 0.2) that creates convex combinations of samples and labels, enhancing generalization.
- **Learning Rate Schedule**: More aggressive learning rate reduction (factor 0.5, patience 2) to better navigate the loss landscape.
- **Adam Optimizer Tuning**: Modified beta_2 parameter to 0.98 for more stable training with WAV2VEC features.
- **Native Keras Format**: Switched to the recommended modern .keras file format for model saving.

### 3. Evaluation Enhancements

- **ROC-AUC Curves**: Added per-class ROC curves with AUC metrics for better model evaluation.
- **Improved History Tracking**: Extended history tracking and visualization, with saved numpy arrays for future analysis.
- **Directory-based Organization**: Better organization of model checkpoints and analysis results with timestamped directories.

## Performance Improvements

The v2 model aims to address several limitations observed in the initial model:

1. **Early Plateauing**: The original model plateaued around epoch 5-10. The learning rate schedule helps push past this plateau.
2. **Class Imbalance Handling**: Improved training process to better handle class imbalance, particularly for "hard" emotion classes.
3. **Overfitting Management**: More aggressive regularization (dropout, noise, MixUp) to manage the high capacity of the model.

## Usage Guide

### Deployment

To deploy and start training the ATTN-CRNN v2 model on the EC2 instance:

```bash
./deploy_attn_crnn_v2.sh
```

This script:
1. Creates necessary directories locally and on the EC2 instance
2. Uploads the training script to the EC2 instance
3. Sets up the training environment
4. Launches training in a tmux session named 'attn_crnn_v2'

### Monitoring

To monitor the training progress:

```bash
# Basic monitoring (status check + recent logs)
./monitor_attn_crnn_v2.sh

# Show only the recent logs
./monitor_attn_crnn_v2.sh -l

# Continuous monitoring (streaming updates)
./monitor_attn_crnn_v2.sh -c
```

The monitoring script provides:
- Training status check (tmux session, process, feature files, model checkpoint)
- Recent training logs (last 50 lines)
- Continuous streaming of important training metrics (epochs, validation results, etc.)

### Model Evaluation

The training script automatically generates:
- Classification report
- Confusion matrix
- Training history plots
- ROC curves for each class

These are saved to the `analysis_results/attn_crnn_v2_TIMESTAMP` directory both locally and on the EC2 instance.

## Hyperparameters

Key hyperparameters that can be adjusted (via command-line arguments):

- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--patience`: Early stopping patience (default: 6)
- `--noise`: Gaussian noise sigma (default: 0.005)
- `--mixup`: MixUp alpha parameter (default: 0.2)
- `--disable_augmentation`: Flag to disable all data augmentation

## Expected Improvements

Based on the implemented enhancements, we expect:

1. Higher overall validation accuracy (target: >60%)
2. Better F1-score for challenging emotion classes (particularly classes 2, 3, and 5)
3. More efficient training with earlier convergence
4. Reduced overfitting observed in the original model
