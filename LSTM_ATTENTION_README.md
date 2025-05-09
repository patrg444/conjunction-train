# LSTM Attention Model for Emotion Recognition

This document details the improvements made to the emotion recognition model and provides instructions for AWS deployment.

## Improvements Made

### 1. Temporal Modeling with Properly Masked BiLSTMs

- Implemented a clean LSTM-based architecture that preserves masking information
- Ensures proper handling of variable-length sequences
- BiLSTM layers capture temporal patterns in audio and video features

### 2. Enhanced Attention Mechanism

- Robust attention layer with explicit mask handling
- Properly handles padding through negative infinity attention scores for padded values
- Re-normalizes weights to ensure valid attention distribution
- Allows the model to focus on important segments of speech/facial expressions

### 3. Data Handling Improvements

- Fixed sequence generators to handle empty batches at end of epochs
- Added fallback values for max() function to prevent empty sequence errors
- Proper tensor shape handling for variable-length inputs
- Ensures training stability across different batch compositions

### 4. Focal Loss with Class Weighting

- Implemented focal loss to focus training on hard examples
- Added class weights to handle class imbalance in the dataset
- Results in more balanced learning across all emotion categories

## AWS Deployment

The model has been set up for deployment on AWS using GPU acceleration for faster training. The following scripts facilitate this process:

### Deployment Scripts

1. **deploy_lstm_attention_model.sh**
   - Deploys the model to an AWS g4dn.xlarge instance (GPU-enabled)
   - Packages and uploads the project files
   - Sets up the required environment with TensorFlow-GPU
   - Starts training in a tmux session for persistence

2. **check_sequence_generator.sh**
   - Verifies that the sequence_data_generator.py on the AWS instance contains all necessary fixes
   - Automatically uploads the fixed version if issues are detected
   - Ensures the deployment has the correct implementation

3. **monitor_lstm_attention_model.sh**
   - Created automatically during deployment
   - Allows real-time monitoring of training logs
   - Includes option to monitor GPU usage with `--gpu` flag

4. **download_lstm_attention_results.sh**
   - Created automatically during deployment
   - Downloads trained models and logs when training completes

### Deployment Process

1. Run the deployment script:
   ```bash
   ./aws-setup/deploy_lstm_attention_model.sh
   ```

2. Verify the sequence generator implementation:
   ```bash
   ./aws-setup/check_sequence_generator.sh
   ```

3. Use continuous monitoring to get a real-time feed of training progress:
   ```bash
   # For training log monitoring (default)
   ./aws-setup/live_continuous_monitor.sh

   # For GPU utilization monitoring
   ./aws-setup/live_continuous_monitor.sh gpu

   # For system resource monitoring
   ./aws-setup/live_continuous_monitor.sh system

   # For comprehensive dashboard with all metrics in tmux panes
   ./aws-setup/live_continuous_monitor.sh all
   ```

4. Download results after training completes:
   ```bash
   ./aws-setup/download_lstm_attention_results.sh
   ```

### AWS Instance Details

- Instance Type: g4dn.xlarge (4 vCPUs, 16GB RAM, 1 NVIDIA T4 GPU)
- AMI: Amazon Linux 2 with NVIDIA drivers
- Training Speed: ~3-4x faster than CPU-only training
- Cost: Approximately $0.50-$0.60 per hour

## Model Architecture

The LSTM-based attention architecture:

```
Audio Input → Masking → BiLSTM(128) → Dropout → BiLSTM(64) → Attention → Features
                                                                               ↓
                                                                           Merge → Dense → Output
                                                                               ↑
Video Input → Masking → BiLSTM(256) → Dropout → BiLSTM(128) → Attention → Features
```

This architecture properly maintains sequence masking information throughout the network, allowing the model to correctly handle variable-length sequences from both audio and video modalities.
