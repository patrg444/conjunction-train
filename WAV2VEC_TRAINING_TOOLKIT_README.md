# Wav2Vec Audio Emotion Recognition Training Toolkit

This toolkit provides a comprehensive set of scripts for training, monitoring, and evaluating wav2vec2-based audio emotion recognition models on AWS EC2 GPU instances.

## Overview

These scripts were created to address the key steps in wav2vec audio emotion training workflows:

1. Training with optimized hyperparameters and mixed precision
2. Real-time monitoring of GPU utilization and training progress
3. TensorBoard visualization setup
4. Model checkpoint and history file downloads

## Scripts

### 1. Training Scripts

- **run_wav2vec_audio_only.sh**: Original training script with default parameters
- **run_wav2vec_audio_only_optimized.sh**: Enhanced training script with:
  - Mixed precision (FP16) enabled for faster training
  - Increased batch size (128)
  - Optimized learning rate (3e-4) 
  - Increased max epochs (200) with early stopping
  
### 2. Monitoring Scripts

- **monitor_wav2vec_training.sh**: Comprehensive monitoring that shows:
  - Training process status
  - GPU utilization metrics
  - Current training progress (epochs, metrics)
  - Error detection in logs
  - Real-time log streaming
  
```bash
./monitor_wav2vec_training.sh <path-to-key.pem> <ec2-ip-address>
```

### 3. TensorBoard Setup

- **setup_tensorboard_tunnel.sh**: Creates an SSH tunnel for TensorBoard visualization
  - Starts TensorBoard on the remote instance if not already running
  - Provides instruction for setting up the SSH tunnel
  - Opens access to the TensorBoard web interface

```bash
./setup_tensorboard_tunnel.sh <path-to-key.pem> <ec2-ip-address>
```

### 4. Model Download

- **download_wav2vec_model.sh**: Downloads trained model files after completion
  - Retrieves best model weights
  - Gets training history for plotting
  - Downloads model architecture files
  - Suggests next steps for evaluation

```bash
./download_wav2vec_model.sh <path-to-key.pem> <ec2-ip-address>
```

## Workflow

1. **Start optimized training**:
   ```bash
   ./run_wav2vec_audio_only_optimized.sh
   ```

2. **Monitor training progress**:
   ```bash
   ./monitor_wav2vec_training.sh ~/Downloads/gpu-key.pem 54.162.134.77
   ```

3. **Set up TensorBoard visualization**:
   ```bash
   ./setup_tensorboard_tunnel.sh ~/Downloads/gpu-key.pem 54.162.134.77
   ```
   Then in your browser, open: `http://localhost:6006`

4. **Download trained model**:
   ```bash
   ./download_wav2vec_model.sh ~/Downloads/gpu-key.pem 54.162.134.77
   ```

5. **Plot training curves**:
   ```bash
   python scripts/plot_training_curve.py --history_file wav2vec_models/<timestamp>_history.json --metric both
   ```

## Key Improvements

1. **Mixed Precision Training**: Enables faster training and better memory utilization by using FP16
2. **Hyperparameter Optimization**: Lower learning rate, larger batch size for better convergence
3. **Comprehensive Monitoring**: Real-time visibility into GPU usage and training progress
4. **Streamlined TensorBoard**: Easy visualization of training metrics and model performance
5. **Automated Downloading**: Simple download of model files for evaluation and deployment

## Requirements

- AWS EC2 instance with GPU (G4dn or G5 recommended)
- SSH key for EC2 instance access
- TensorFlow 2.x on the EC2 instance
- Prepared wav2vec feature files in the correct directory structure
