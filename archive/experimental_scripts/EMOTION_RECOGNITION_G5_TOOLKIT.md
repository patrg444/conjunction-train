# Emotion Recognition G5 Training Toolkit

A comprehensive toolkit to fix and optimize emotion recognition model training on AWS G5 GPU instances. This toolkit addresses issues with missing data, normalization, and monitoring in deep learning pipelines.

## 📋 Problem Overview

The current training run on the G5 instance has the following issues:

- ❌ Missing RAVDESS and CREMA-D feature archives (falling back to dummy data)
- ❌ Placeholder laughter manifest with only ~20 entries 
- ❌ Missing normalization statistics
- ❌ Low GPU utilization (≈0%)

## 🛠️ Toolkit Components

| Script | Purpose |
|--------|---------|
| `fix_and_restart_g5_training.sh` | **One-step solution** that combines all fixes and restarts training |
| `generate_laughter_manifest.py` | Creates a proper laughter detection manifest |
| `fix_normalization_stats.py` | Generates/fixes audio and video normalization stats |
| `fix_g5_training.sh` | Core script to upload and extract feature archives |
| `continuous_g5_monitor.sh` | Advanced real-time monitoring with color-coded metrics |

## 🚀 Quick Start

For an all-in-one fix:

```bash
# 1. Edit to set your S3 bucket and check paths
nano fix_and_restart_g5_training.sh

# 2. Run the one-step fix script
./fix_and_restart_g5_training.sh

# 3. Monitor the training progress
./continuous_g5_monitor.sh
```

## 📊 Expected Results

After successfully applying the fixes:

- ✅ RAVDESS dataset: ~1.6 GB present on EC2
- ✅ CREMA-D dataset: ~1.0 GB present on EC2
- ✅ Laughter manifest: 500+ entries with proper train/val/test split
- ✅ Normalization statistics: audio_normalization_stats.pkl and video_normalization_stats.pkl present
- ✅ GPU utilization: 60-90% (using the GPU effectively)
- ✅ Training progress: normal epoch times (5-10 minutes per epoch)

## 🔍 Detailed Usage

### 1. Generate Laughter Manifest

This creates a realistic manifest file with customizable parameters:

```bash
./generate_laughter_manifest.py --output datasets/manifests/laughter_v1.csv --samples 500
```

### 2. Fix Normalization Stats

Computes and saves normalization statistics from feature files:

```bash
./fix_normalization_stats.py
```

### 3. Single Step Fix

The comprehensive fix script that:
- Stops current training
- Uploads feature archives via S3
- Fixes normalization
- Restarts training

```bash
./fix_and_restart_g5_training.sh
```

### 4. Enhanced Monitoring

Continuous real-time monitoring with:
- Color-coded metrics
- GPU utilization tracking
- Dataset size verification
- Training progress with ETA
- TensorBoard integration

```bash
./continuous_g5_monitor.sh [interval_seconds]
```

## 📝 Implementation Notes

- The generator looks for FaceNet features under `~/emotion-recognition/ravdess_features_facenet/` and `~/emotion-recognition/crema_d_features_facenet/`
- If those folders are absent or <1MB, the generator prints warnings and builds 10 random dummy samples
- The training script expects a manifest at `datasets/manifests/laughter_v1.csv`
- Normalization pickles should be in `models/dynamic_padding_no_leakage/`

## 🧰 Prerequisites

Before using this toolkit, ensure you have:

- AWS CLI configured with access to your S3 bucket
- SSH access to the G5 instance (key in `~/Downloads/gpu-key.pem`)
- The feature archives prepared or available for upload
