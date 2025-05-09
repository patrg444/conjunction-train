# G5 Training Solution: Using Real Data Instead of Dummy Samples

## Problem Overview

The G5 instance training was falling back to dummy data instead of using the full RAVDESS + CREMA-D + Humor/Laughter datasets, resulting in:

- **GPU utilization ≈ 0%** (should be 60-90%)
- Warning messages: "Data size smaller than expected or not present"
- Training with tiny dummy datasets (10 samples per split)
- Fast but meaningless epoch completion
- Poor model quality

## Root Causes Identified

1. **Missing feature archives**: Required pre-extracted NPZ files were not present on the EC2 instance
2. **Missing normalization pickles**: `audio_normalization_stats.pkl` and `video_normalization_stats.pkl` absent
3. **Incomplete laughter manifest**: Only a placeholder with ~20 dummy entries instead of hundreds

## Solution Components Created

### 1. Core Fix Scripts

- `fix_g5_training.sh`: Core script that downloads and extracts feature archives from S3
- `fix_and_restart_g5_training.sh`: All-in-one solution that combines all fixes
- `fix_normalization_stats.py`: Generates missing normalization pickles
- `generate_laughter_manifest.py`: Creates a proper 500-entry manifest for laughter detection

### 2. Monitoring Tools

- `continuous_g5_monitor.sh`: Real-time monitoring with color-coded metrics
- `setup_tensorboard_tunnel.sh`: Sets up TensorBoard visualization via SSH tunnel
- `download_g5_model.sh`: Downloads the trained model and logs after completion
- `verify_s3_assets.sh`: Verifies the presence of required archives in S3

### 3. Documentation

- `G5_TRAINING_EXECUTION_GUIDE.md`: Step-by-step instructions
- `G5_TRAINING_FIX_README.md`: Detailed problem/solution description
- `EMOTION_RECOGNITION_G5_TOOLKIT.md`: Overview of all toolkit components

## How to Use the Solution

### Step 1: Verify S3 Assets

Run the verification script to ensure the feature archives are available in S3:

```bash
./verify_s3_assets.sh
```

If archives are missing, upload them:

```bash
aws s3 cp ravdess_features_facenet.tar.gz s3://emotion-recognition-data/
aws s3 cp crema_d_features_facenet.tar.gz s3://emotion-recognition-data/
```

### Step 2: Apply the Fix

Run the all-in-one fix script:

```bash
./fix_and_restart_g5_training.sh
```

This script will:
- Stop any existing training processes
- Download feature archives from S3 (~2.6GB total)
- Extract them to the correct locations
- Create/verify the laughter manifest with 500 entries
- Fix normalization statistics
- Restart training with real data

### Step 3: Monitor Training Progress

Run the monitoring script to ensure training is using real data:

```bash
./continuous_g5_monitor.sh 15  # Updates every 15 seconds
```

You should see:
- RAVDESS features: ~1.6GB
- CREMA-D features: ~982MB
- Laughter manifest: 500+ entries
- Normalization files: 2
- GPU utilization: 60-90%
- Training making actual progress with real data

### Step 4: Visualize Training (Optional)

Set up TensorBoard for visualizing training metrics:

```bash
./setup_tensorboard_tunnel.sh
```

This will open TensorBoard in your browser at `http://localhost:6006`.

### Step 5: Download Trained Model

After training completes (~8-10 hours for 100 epochs), download the model:

```bash
./download_g5_model.sh
```

## Expected Results

- Full-scale training with real data instead of dummy samples
- GPU utilization in the 60-90% range
- Proper learning, with validation accuracy improving over time
- Training time of ~5-8 minutes per epoch (rather than seconds)
- Final model with significantly better performance

## Troubleshooting

If issues persist:

1. Verify S3 bucket permissions and EC2 instance IAM role
2. Check SSH connectivity and key permissions
3. Inspect the feature directories on EC2 for proper extraction
4. Manually verify the laughter manifest has 500+ entries
5. Ensure normalization stats files are in place

For detailed diagnostics, check the training log:

```bash
ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "tail -n 100 ~/emotion-recognition/logs/train_laugh_*.log"
```

## Completion Verification

To verify the fix was successful, look for these key indicators:

1. Directory sizes: 
   - RAVDESS ~1.6GB
   - CREMA-D ~1.0GB
   
2. GPU utilization consistently above 60%

3. Training progress indicators:
   - Reasonable time per epoch (minutes, not seconds)
   - Improving accuracy metrics
   - Training loss decreasing over time
