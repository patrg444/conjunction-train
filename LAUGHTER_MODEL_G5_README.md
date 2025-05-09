# Audio-Video-Laughter Model Training on AWS G5 GPU

This document outlines the entire process for training the 100-epoch audio-video-laughter emotion recognition model on AWS G5.2xlarge GPU instance.

## Overview

We've successfully deployed code and training configuration to an AWS G5.2xlarge instance running Ubuntu. The training will use synchronized audio-video feature data with laughter recognition capabilities running for 100 epochs with a batch size of 256 and MAX_SEQ_LEN=45 (compatible with the realtime inference window).

## Setup Summary

1. **SSH Connectivity Issue Fixed**:
   - Previously, SSH authentication was failing due to a key mismatch
   - Created a new key pair (`gpu-key`) and launched a new G5.2xlarge instance
   - Updated scripts to use the correct key path and IP address
   - Verified SSH connectivity

2. **Deployment Status**:
   - Code and normalization statistics successfully deployed
   - Training launcher script created on EC2 instance
   - Download script generated locally

3. **Training Launch**:
   - The automated workflow encountered an issue during data presence verification
   - Created custom scripts to properly launch and monitor the training

## Available Scripts

| Script | Purpose |
|--------|---------|
| `start_training_20250419_131546.sh` | Starts the training job on the EC2 instance |
| `monitor_training_20250419_131515.sh` | Sets up monitoring daemon on EC2 and displays real-time logs |
| `download_g5_model_20250419_131350.sh` | Downloads the trained model after completion |

## Workflow Instructions

### 1. Launch Training

```bash
./start_training_20250419_131546.sh
```

This will launch a full 100-epoch training job with the following parameters:
- Batch Size: 256
- MAX_SEQ_LEN: 45 (compatible with realtime inference)
- Laughter Loss Weight: 0.3

### 2. Monitor Progress

```bash
./monitor_training_20250419_131515.sh
```

This will:
- Set up a monitoring daemon on the EC2 instance if not already running
- Display real-time logs including training metrics
- Show GPU utilization statistics

### 3. Download Model After Completion

```bash
./download_g5_model_20250419_131350.sh
```

This will:
- Check if training is complete
- Download the best model checkpoint
- Copy evaluation metrics
- Generate a local summary of model performance

## Expected Timeline

- Training Duration: ~10 hours for 100 epochs
- Validation every 5 epochs
- Model will be stored in `models/audio_pooling_with_laughter_*` with timestamp

## Post-Training Verification

After downloading the model, run:

```bash
python scripts/demo_emotion_with_laughter.py --model models/audio_pooling_with_laughter_*/model_best.h5
```

## Bugfix and Current Status

We identified and fixed a critical issue in the feature_normalizer.py module:
- Problem: `ImportError: cannot import name 'load_normalization_stats'` when trying to start training
- Fix: Added the missing `load_normalization_stats` function (alias for `load_or_estimate_normalization_stats`)
- Solution implemented in `fixed_feature_normalizer.py` and deployed to EC2

The G5.2xlarge GPU instance is now ready for training with:
- Public IP: 18.208.166.91
- All code and normalization statistics deployed
- Fixed feature_normalizer.py uploaded
- Training ready to restart

To start the training process:
```bash
./start_training_20250419_131546.sh
```

To monitor the progress once training has started:
```bash
./monitor_training_20250419_131515.sh
```
