# Training Monitor for Emotion Recognition Model

This document explains how to monitor the training progress of the emotion recognition model that uses both RAVDESS and CREMA-D datasets.

## What We've Accomplished

1. **Successfully fixed training script** to properly load and use both datasets:
   - RAVDESS dataset (1,440 samples) organized in Actor folders
   - CREMA-D dataset (7,440 samples)
   - Combined total: 8,880 samples for training

2. **Created monitoring tools** to track training progress in real-time

3. **Implemented a branched neural network** that processes:
   - Audio features (89 dimensions)
   - Video/facial features (512 dimensions from FaceNet)

## Monitoring Options

We've created two scripts to help you monitor the training progress:

### 1. Basic Live Monitor (`live_monitor.sh`)

This is a simple script that continuously streams the training log with colored highlighting for important information.

```bash
cd aws-setup
./live_monitor.sh
```

- Provides raw output from the training log
- Highlights key information like epochs, accuracy, and errors
- Press Ctrl+C to exit

### 2. Enhanced Monitor (`enhanced_monitor.sh`)

This script provides a better formatted view with regular updates and summarized information.

```bash
cd aws-setup
./enhanced_monitor.sh
```

- Shows dataset information (RAVDESS/CREMA-D sample counts)
- Displays current epoch, training and validation metrics
- Automatically updates every 5 seconds
- Highlights any errors or warnings
- Press Ctrl+C to exit

## Training Details

The model is training with the following configuration:

- **Architecture**: Branched neural network (audio + video streams)
- **Batch size**: 32
- **Epochs**: 20
- **Patience**: 5 (early stopping)
- **Sequence length**: 50 frames
- **Classes**: 6 emotions (anger, disgust, fear, happy, neutral, sad)
- **Train/Val split**: 80%/20% (7,104 training samples, 1,776 validation samples)

## Expected Results

Training is currently in progress and is expected to achieve:

- **Training time**: ~2 hours on CPU
- **Expected accuracy**: 45-60% on validation set
- **Output files**:
  - Best model: `models/combined/model_best.h5`
  - Final model: `models/combined/final_model.h5`

## Interpreting Results

- The model should show increasing validation accuracy over time
- The combined dataset should provide better generalization than either dataset alone
- The model might achieve better accuracy on some emotions than others

## Troubleshooting

If you encounter issues with the monitoring scripts:

1. Ensure SSH key permissions are correct: `chmod 400 aws-setup/emotion-recognition-key-*.pem`
2. Check if the EC2 instance is still running: `cd aws-setup && ./monitor_cpu.sh`
3. Verify the training process is active: `cd aws-setup && ./check_progress.sh`
