# LSTM Attention Model Deployment Guide

## Overview

This document describes the deployment of the LSTM Attention Model for emotion recognition on an existing AWS EC2 instance. The model uses facial feature data from the RAVDESS and CREMA-D datasets to predict emotions with an attention mechanism that helps the model focus on the most relevant parts of the temporal sequences.

## Deployment Summary

| Component | Value |
|-----------|-------|
| Instance ID | i-0dd2f787db00b205f |
| Instance Type | c5.24xlarge (96 vCPUs) |
| Instance IP | 98.82.121.48 |
| SSH Key | emotion-recognition-key-20250322082227.pem |
| Training Script | train_branched_attention.py |
| Environment | Conda (Python 3.8) |
| Model Output | ~/emotion_training/models/attention_focal_loss/ |

## Deployment Process

We performed the following steps to deploy the LSTM attention model:

1. **Identified an existing EC2 instance** - Found a running c5.24xlarge instance that had previously been used for emotion recognition training.

2. **Uploaded required scripts** - Transferred the necessary training scripts to the instance:
   - `train_branched_attention.py` - Main training script for the LSTM attention model 
   - `sequence_data_generator.py` - Data generator for handling sequential data

3. **Environment Setup** - Created a Conda environment with the required dependencies:
   - Python 3.8
   - TensorFlow 2.9.0
   - NumPy, Pandas, SciPy, Scikit-learn
   - Matplotlib and H5py

4. **Model Training** - Started the training process in a detached tmux session for persistence, allowing the training to continue even if the SSH connection is closed.

## Key Features of the LSTM Attention Model

- **Bidirectional LSTM layers** - Captures both past and future temporal information
- **Self-attention mechanism** - Helps the model focus on the most important frames in the sequence
- **Dynamic sequence handling** - Properly handles variable-length input sequences
- **Combined datasets** - Trained on both RAVDESS and CREMA-D datasets for better generalization
- **Focal loss function** - Gives more weight to difficult-to-classify examples

## Monitoring Tools

We've created several scripts to monitor the training process:

1. **check_training_files.sh** - Shows directory structure, available scripts, and recent log entries:
   ```bash
   bash aws-setup/check_training_files.sh
   ```

2. **stream_logs.sh** - Streams the training logs in real-time:
   ```bash
   bash aws-setup/stream_logs.sh
   ```

3. **monitor_cpu_usage.sh** - Monitors CPU usage and active processes:
   ```bash
   bash aws-setup/monitor_cpu_usage.sh
   ```

4. **monitor_running_job.sh** - Interactive monitoring with multiple options:
   ```bash
   bash aws-setup/monitor_running_job.sh
   ```

## Expected Training Time

Training the LSTM attention model on the c5.24xlarge instance is expected to take approximately 8-12 hours depending on the dataset size and model hyperparameters.

## Accessing Trained Models

Once training is complete, the model will be saved to:
```
~/emotion_training/models/attention_focal_loss/final_model.h5
```

You can download the trained model using:
```bash
bash aws-setup/download_results.sh
```

## Troubleshooting

If the training process stops unexpectedly, you can:

1. Check the logs for errors:
   ```bash
   bash aws-setup/stream_logs.sh
   ```

2. Restart the training:
   ```bash
   bash aws-setup/conda_environment_fix.sh
   ```

3. If the Conda environment is corrupted, you can rebuild it:
   ```bash
   bash aws-setup/conda_environment_fix.sh
   ```
