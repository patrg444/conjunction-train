# LSTM Attention Model Without Audio Augmentation

This documentation outlines the implementation of an LSTM-based attention model for multimodal emotion recognition without audio augmentation. This version addresses a fundamental issue in the previous implementation where audio augmentation created misalignment between audio and video modalities.

## Problem: Audio-Visual Misalignment in Previous Implementation

The original LSTM attention model used audio augmentation techniques including:
- Noise addition
- Pitch/speaker variation
- Tempo variation through resampling

These augmentation techniques modified the audio features while keeping the corresponding video features unchanged. This created a fundamental misalignment between the two modalities, particularly with tempo variation which changes the temporal structure of the audio sequence.

The misalignment is problematic because:
1. In real-world scenarios, audio and video cues are naturally synchronized (e.g., lip movements match speech sounds)
2. The model may learn incorrect relationships between modalities
3. Temporal attention mechanisms may struggle to find meaningful correlations across modalities
4. The natural audio-visual synchronization that exists in real data is broken

## Solution: Remove Audio Augmentation

The solution implemented here removes all audio augmentation while preserving other key aspects of the model:
- Temporal attention layers after LSTMs
- Focal loss for handling class imbalance 
- Class weighting for imbalanced emotion labels
- Proper data normalization to prevent data leakage

This approach ensures that the audio and video modalities remain properly aligned during training, which should lead to more consistent and interpretable model behavior.

## Key Files

### Training Script
- **scripts/train_branched_attention_no_aug.py**: Modified version of the training script with audio augmentation disabled.

### Deployment Scripts
- **aws-setup/deploy_lstm_attention_no_aug.sh**: Deploys the training script to an EC2 instance
- **aws-setup/download_lstm_attention_no_aug_results.sh**: Downloads trained models from the EC2 instance
- **aws-setup/terminate_lstm_attention_no_aug_instance.sh**: Terminates the EC2 instance after training

## Usage Instructions

### 1. Deploy to EC2 for Training

```bash
./aws-setup/deploy_lstm_attention_no_aug.sh
```

This script:
- Creates a c5.24xlarge EC2 instance with 96 vCPUs
- Uploads the necessary scripts and data
- Configures the environment and dependencies
- Starts the training process in the background

### 2. Monitor Training

You can monitor the training progress using:

```bash
./aws-setup/continuous_training_monitor.sh
```

### 3. Download Results

Once training is complete or you want to check the latest model:

```bash
./aws-setup/download_lstm_attention_no_aug_results.sh
```

This downloads:
- Best model (saved based on validation accuracy)
- Final model (at the end of training)
- Training log with metrics

### 4. Terminate EC2 Instance

When you're done with training, terminate the instance to avoid additional charges:

```bash
./aws-setup/terminate_lstm_attention_no_aug_instance.sh
```

## Expected Benefits

This implementation should provide several advantages:

1. **Better Feature Correlation**: The model will learn from properly synchronized audio-visual data, allowing it to discover meaningful correlations between modalities.

2. **More Accurate Attention Mechanisms**: The attention layers will be able to focus on truly relevant parts of sequences across both modalities.

3. **Improved Interpretability**: Without misalignment, the attention weights should be more interpretable, showing which parts of the synchronized sequences are most relevant for emotion recognition.

4. **Potentially Higher Accuracy**: By learning from natural audio-visual correspondences, the model may achieve better performance on real-world data where modalities are naturally aligned.

## Future Improvements

While removing audio augmentation addresses the alignment issue, a more sophisticated approach for future work could include:

1. **Synchronized Augmentation**: Apply corresponding transformations to both modalities simultaneously (e.g., if audio is sped up, video should be as well).

2. **Alignment-Preserving Augmentation**: Focus only on augmentations that preserve alignment (like adding noise to audio and slight transformations to video).

3. **Feature-Level Augmentation**: Apply augmentation at feature level rather than temporal level, to avoid disrupting temporal alignment.
