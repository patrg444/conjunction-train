# Hybrid Conv1D-TCN Model with Cross-Modal Attention

This document describes the hybrid emotion recognition model that combines Conv1D for audio processing, TCN with self-attention for video, and cross-modal attention fusion.

## Architecture Overview

This advanced architecture improves upon previous models by using specialized approaches for each modality:

### Audio Branch
- **Conv1D layers** instead of LSTM for audio processing
- More efficient for spectral patterns and feature extraction
- Multi-scale temporal feature learning (3×3 and 5×5 kernels)
- Lower computational cost than LSTM

### Video Branch
- **Temporal Convolutional Network (TCN)** with exponentially dilated convolutions
- Self-attention mechanism for global temporal context
- More effective at modeling long-range dependencies in facial expressions
- Better computational efficiency than bidirectional LSTM

### Cross-Modal Fusion
- Bidirectional cross-modal attention
- Audio features attend to video features and vice versa
- Learns modality-specific importance weighting
- Allows each modality to focus on relevant parts of the other

## Expected Improvements

Compared to the LSTM-based models:
- Lower memory usage and faster training times
- Better computational efficiency on CPU instances
- More effective audio feature extraction with Conv1D
- Improved handling of long-range dependencies in video with TCN
- Enhanced fusion through cross-modal attention

## Usage Instructions

### Deployment
To deploy the model to the EC2 instance:
```bash
./aws-setup/deploy_hybrid_attention.sh
```
This script will:
1. Terminate the poorly performing RL model
2. Upload the hybrid model script
3. Launch the training process on the EC2 instance

### Monitoring
To monitor training progress:
```bash
./aws-setup/monitor_hybrid_attention.sh
```
This interactive script provides options to:
- View training logs
- Stream logs in real-time
- Check system resource usage
- View validation metrics
- Estimate training completion time

### Downloading Results
When training is complete, download the model and logs:
```bash
./aws-setup/download_hybrid_attention_results.sh
```
This will:
- Download the trained model files
- Download and analyze training logs
- Create summary files with key metrics

## Implementation Details

- Based on `train_branched_attention_no_aug.py` with major architecture changes
- Uses the same preprocessing and data loading pipeline
- Maintains dynamic padding for variable-length sequences
- Keeps focal loss for handling class imbalance
- Same training parameters (batch size, epochs, learning rate)

## Model Parameters

- Batch size: 24
- Max epochs: 50
- Initial learning rate: 0.0005
- Early stopping patience: 10
- Focal loss gamma: 2.0
- Class weights: Inverse frequency weighting
