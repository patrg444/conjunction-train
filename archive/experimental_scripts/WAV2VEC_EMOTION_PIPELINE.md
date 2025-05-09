# Wav2Vec Emotion Recognition Pipeline

This document provides a complete guide to the wav2vec-based emotion recognition pipeline, from feature extraction to model training and fusion.

## Overview

This pipeline enables you to:

1. Extract wav2vec features from RAVDESS and CREMA-D video datasets
2. Train neural network models (LSTM or Transformer) on these features
3. Create a fusion model combining audio (wav2vec) and video (SlowFast) features

## Prerequisites

- Access to the EC2 instance with GPU support
- The RAVDESS and CREMA-D datasets already uploaded to the EC2 instance at:
  - `/home/ubuntu/datasets/ravdess_videos/` (MP4 files)
  - `/home/ubuntu/datasets/crema_d_videos/` (FLV files)
- Python with PyTorch and TensorFlow installed on the EC2 instance

## Pipeline Steps

### 1. Extract wav2vec Features

Two scripts are provided for feature extraction:

- **`deploy_enhanced_wav2vec_extraction_to_ec2.sh`**: Uploads and runs the extraction script with a small sample
- **`extract_full_wav2vec_datasets.sh`**: Processes the complete datasets in the background

To extract features from a sample of videos:

```bash
./deploy_enhanced_wav2vec_extraction_to_ec2.sh
```

To extract features from all videos in the background:

```bash
./extract_full_wav2vec_datasets.sh
```

This creates NPZ files with wav2vec features in the `models/wav2vec/` directory on EC2.

### 2. Train Audio Emotion Model

After feature extraction, you can train a neural network model on the extracted features:

```bash
./train_wav2vec_emotion_model.sh
```

This script:
1. Uploads training scripts to EC2
2. Provides instructions for starting training
3. Supports both LSTM and Transformer architectures:
   - `./train_wav2vec_emotion.sh lstm 32 50 0.001`
   - `./train_wav2vec_emotion.sh transformer 32 50 0.001`

The trained model will be saved to `models/audio_emotion/` on EC2.

### 3. Create Fusion Model

Finally, you can create a fusion model that combines the audio model with a pre-trained SlowFast video model:

```bash
./create_emotion_fusion_model.sh
```

This script:
1. Uploads fusion model scripts to EC2
2. Provides instructions for creating the fusion model:
   - `./create_fusion_model.sh 0.7 0.3` (70% video, 30% audio weight)
3. Creates a fusion configuration and demo script

The fusion model will be saved to `models/fusion/` on EC2.

## Downloading Results

Each script provides instructions for downloading results:

- **Features**: 
  ```bash
  mkdir -p wav2vec_extracted_features
  scp -i ~/Downloads/gpu-key.pem -r ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/models/wav2vec/ ./wav2vec_extracted_features/
  ```

- **Audio Model**:
  ```bash
  mkdir -p wav2vec_emotion_model
  scp -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/wav2vec_emotion_model.tar.gz wav2vec_emotion_model/
  tar -xzvf wav2vec_emotion_model/wav2vec_emotion_model.tar.gz -C wav2vec_emotion_model/
  ```

- **Fusion Model**:
  ```bash
  mkdir -p fusion_model
  scp -i ~/Downloads/gpu-key.pem ubuntu@54.162.134.77:/home/ubuntu/audio_emotion/fusion_model.tar.gz fusion_model/
  tar -xzvf fusion_model/fusion_model.tar.gz -C fusion_model/
  ```

## Performance and Results

The initial test extraction successfully:
- Processed 8 RAVDESS MP4 files (2 skipped due to "surprised" emotion not in our class map)
- Processed 10 CREMA-D FLV files
- Created fusion configuration with 6 emotion classes: angry, disgust, fearful, happy, neutral, sad

Dataset statistics:
- RAVDESS: 1,440 MP4 files
- CREMA-D: 7,442 FLV files

## Troubleshooting

- **File Format Issues**: The pipeline now supports both MP4 and FLV formats
- **Memory Errors**: Use smaller batch sizes or reduce sequence length in the training scripts
- **Missing Files**: Check paths and ensure datasets are correctly uploaded
- **CUDA Errors**: Ensure GPU memory growth is enabled (included in scripts)

## Extensions

The pipeline can be extended by:
1. Adding more datasets
2. Tuning hyperparameters
3. Implementing different model architectures
4. Integrating with real-time inference systems
