# SlowFast Emotion Recognition Workflow

This document explains the complete workflow for deploying and training the SlowFast-R50 based emotion recognition model using the RAVDESS and CREMA-D datasets already uploaded to the EC2 instance.

## Overview

The SlowFast model is a state-of-the-art architecture for video action recognition, modified here for facial emotion recognition. This implementation is expected to improve accuracy from ~60% to ~70% compared to simpler R3D-18 models.

## Files and Components

### Core Components
- `config/slowfast_face.yaml`: Configuration file with hyperparameters
- `scripts/train_slowfast_emotion.py`: The main training script
- `scripts/generate_video_manifest.py`: Creates dataset manifest from video files

### Deployment Scripts
- `deploy_complete_slowfast_pipeline.sh`: All-in-one deployment solution
- `generate_manifest_on_ec2.sh`: Generates manifest file on EC2
- `deploy_slowfast_training.sh`: Deploys and starts training
- `download_slowfast_model.sh`: Downloads trained model from EC2

### Monitoring Scripts
- `monitor_slowfast_progress.sh`: Basic color-coded training progress monitor
- `realtime_slowfast_monitor.sh`: Direct tmux session attachment for true real-time monitoring
- `stream_slowfast_monitor.sh`: Non-interactive streaming output without requiring tmux controls

### Management Scripts
- `check_training_status.sh`: Checks status of running training jobs
- `stop_and_clean_training.sh`: Stops active jobs and cleans up

## Workflow Steps

### 1. Check Training Status
Before starting new training, check if any existing jobs are running:
```bash
./check_training_status.sh
```
This will show active tmux sessions, GPU usage, and running Python processes.

### 2. Based on the status check, either:

#### a) Stop existing training (if needed)
If you need to stop existing training:
```bash
./stop_and_clean_training.sh
```
This will kill active tmux sessions, stop Python processes, and clean up temporary files.

#### b) Let existing training finish
If existing training is nearly complete, monitor it:
```bash
ssh -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77 "tmux attach -t SESSION_NAME"
```
(Replace SESSION_NAME with the actual session name from the status check)

### 3. Deploy the SlowFast Pipeline

Once the EC2 instance is ready, you can run the complete pipeline:
```bash
./deploy_complete_slowfast_pipeline.sh
```

This script will:
1. Verify datasets on EC2
2. Generate the video manifest file
3. Copy SlowFast training files to EC2
4. Prepare the training environment
5. Launch training in a tmux session
6. Set up monitoring

### 4. Monitor Training Progress

You have three options for monitoring training progress, each with different capabilities:

#### Basic Monitoring (color-coded output with periodic updates)
```bash
./monitor_slowfast_progress.sh
```
This provides basic color-coded output with progress bars but may have occasional delays.

#### Real-time Interactive Monitoring (full tmux attachment)
```bash
./realtime_slowfast_monitor.sh
```
This attaches directly to the tmux session running the training, providing true real-time monitoring with no buffering. Use Ctrl+B then D to detach without stopping training.

#### Stream Monitoring (non-interactive continuous stream)
```bash
./stream_slowfast_monitor.sh
```
This provides a continuous stream of training output without requiring tmux controls. It shows both previous output and real-time streaming. Press Ctrl+C to stop monitoring (training will continue).

### 5. Download Trained Model

When training is complete:
```bash
./download_slowfast_model.sh
```

## Technical Details

### Manifest File
The manifest file organizes videos from RAVDESS and CREMA-D with:
- File paths to videos
- Emotion labels
- Dataset source (ravdess/crema)
- Train/val/test splits

### SlowFast Architecture
The SlowFast-R50 architecture uses:
- Dual pathways: slow (temporal) and fast (spatial)
- Squeeze-and-Excitation blocks
- Advanced data augmentation techniques
- Multi-clip training and test-time augmentation

## Decision Guide

- For new training, use the complete pipeline (`deploy_complete_slowfast_pipeline.sh`)
- To check existing jobs, use `check_training_status.sh`
- To manage active sessions, use `stop_and_clean_training.sh`
- For just monitoring, use `monitor_slowfast_progress.sh`

The implementation leverages the RAVDESS and CREMA-D datasets that have been uploaded to the EC2 instance, creating a complete emotion recognition solution with improved accuracy.
