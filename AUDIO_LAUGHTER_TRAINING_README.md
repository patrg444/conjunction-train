# Audio Emotion with Laughter Detection - Training Guide

This document outlines the end-to-end workflow for deploying, training, monitoring, and evaluating the multimodal emotion recognition model with laughter detection support.

## Overview

The system uses an audio+video multimodal approach with a branched architecture that:
- Processes facial features (FaceNet embeddings) and audio features
- Detects 8 emotions (primary task)
- Detects laughter presence (auxiliary task)
- Maintains compatibility with real-time inference (MAX_SEQ_LEN=45)

## Prerequisites

- SSH access to AWS instance (private key configured)
- AWS g5.2xlarge instance running (IP: 52.90.38.179)
- Feature archives prepared locally:
  - `ravdess_features_facenet/` (~1.6 GB)
  - `crema_d_features_facenet/` (~1.0 GB)
- Normalization statistics:
  - `models/dynamic_padding_no_leakage/audio_normalization_stats.pkl`
  - `models/dynamic_padding_no_leakage/video_normalization_stats.pkl`

## Training Parameters

- Epochs: 100
- Batch size: 256 
- Max sequence length: 45 (compatible with realtime window)
- Laughter loss weight: 0.3

## Agentic Workflow

We've created a fully automated workflow script that handles the entire process:

```bash
./agentic_train_laughter_g5.sh
```

This script:

1. **Verifies SSH access** to the AWS instance
2. **Checks data presence** on EC2 (feature archives, normalization stats)
3. **Deploys code and data** if needed
4. **Verifies deployment** success with data size checks
5. **Starts training** on EC2
6. **Sets up monitoring daemon** for ongoing progress tracking
7. **Creates monitoring helper scripts**:
   - `agentic_monitor_<timestamp>.sh`: View real-time training logs
   - `agentic_download_<timestamp>.sh`: Download and evaluate the trained model

## Manual Process (if preferred)

If you prefer to run steps manually:

1. Deploy to AWS:
   ```bash
   ./deploy_to_aws_g5.sh
   ```

2. SSH into the instance:
   ```bash
   ssh -i ~/.ssh/id_rsa ubuntu@52.90.38.179
   ```

3. Start training:
   ```bash
   cd ~/emotion-recognition
   ./train_g5_<timestamp>.sh
   ```

4. Monitor training:
   ```bash
   # On EC2:
   tail -f logs/train_laugh_*.log
   nvidia-smi --query-gpu=utilization.gpu --loop=60
   ```

5. Download model after completion:
   ```bash
   # On local machine:
   ./download_g5_model_<timestamp>.sh
   ```

## Validation & Testing

After training completes:

1. Download the model:
   ```bash
   ./agentic_download_<timestamp>.sh
   ```

2. Run local demo:
   ```bash
   python scripts/demo_emotion_with_laughter.py \
     --model models/audio_pooling_with_laughter_*/model_best.h5
   ```

3. Evaluate against previous models:
   ```bash
   python extract_all_models_val_accuracy.py
   python compare_model_accuracies.py --latest
   ```

## Expected Outcomes

- Training duration: ~10 hours on g5.2xlarge
- GPU utilization: 80-90%
- Target validation accuracy: >84% on emotion classification
- Laughter detection F1 score: >0.75

## Troubleshooting

- **SSH connectivity issues**: Ensure correct SSH key path in `agentic_train_laughter_g5.sh`
- **Deployment failure**: Check disk space on EC2 with `df -h`
- **Training crashes**: Examine logs in `~/emotion-recognition/logs/`
- **GPU underutilization**: Increase batch size if GPU memory allows

## Post-Training

After successful training and validation:
1. Update production models with the new checkpoint
2. Update normalization statistics if needed
3. Test in realtime with `scripts/enhanced_compatible_realtime_emotion.py`

## Repository Structure

```
├── deploy_to_aws_g5.sh              # Deployment script
├── agentic_train_laughter_g5.sh     # Automated workflow
├── agentic_monitor_<timestamp>.sh   # Generated monitoring script
├── agentic_download_<timestamp>.sh  # Generated download script
├── download_g5_model_<timestamp>.sh # Model download script
├── models/                          # Trained models directory
│   └── audio_pooling_with_laughter_<timestamp>/
│       ├── model_best.h5            # Best checkpoint
│       ├── training_history.json    # Training metrics
│       └── model_info.json          # Model metadata
└── scripts/
    ├── train_audio_pooling_lstm_with_laughter.py  # Training script
    └── demo_emotion_with_laughter.py              # Demo script
