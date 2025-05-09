# G5 Training Data Fix

This README provides a comprehensive guide to fix the current G5 training issue where the training is running on dummy data rather than the full RAVDESS and CREMA-D datasets.

## Problem Description

The current G5 instance training is **not using the real datasets**:

- The feature archives (RAVDESS/CREMA-D) are missing on the EC2 instance
- The training is falling back to the generator's dummy data (10 samples per split)
- The laughter manifest is just a placeholder with dummy entries
- GPU utilization is near 0% due to minimal data processing

Expected sizes after fixes:
- RAVDESS ≈ 1.6 GB
- CREMA-D ≈ 1.0 GB

## Solution Overview

1. Generate a proper laughter manifest
2. Upload the feature archives to S3
3. Transfer the archives to the EC2 instance
4. Extract the archives in the correct locations
5. Ensure normalization stats are present
6. Restart the training with real data

## Step 1: Generate Laughter Manifest

The model is designed to perform emotion recognition and laughter detection simultaneously. The manifest file maps filenames to laughter labels.

```bash
# Generate a realistic laughter manifest with 500 samples
python generate_laughter_manifest.py --output datasets/manifests/laughter_v1.csv --samples 500
```

This creates a CSV file with format:
```
filepath,laugh,split
Actor_01/01-01-01-01-01-01-01.wav,0,train
CremaD_0001_IEO.wav,1,val
...
```

## Step 2: Upload Feature Archives to S3

You'll need to provide the feature archives (as tar.gz files) for upload:

1. Prepare the feature archives:
   ```bash
   # If you have the extracted feature directories
   tar -czf ravdess_features_facenet.tar.gz ravdess_features_facenet/
   tar -czf crema_d_features_facenet.tar.gz crema_d_features_facenet/
   ```

2. Upload to S3:
   ```bash
   # Replace your-s3-bucket with your actual bucket name
   aws s3 cp ravdess_features_facenet.tar.gz s3://your-s3-bucket/
   aws s3 cp crema_d_features_facenet.tar.gz s3://your-s3-bucket/
   ```

## Step 3: Edit the Script Configuration

Edit `fix_g5_training.sh` to set your S3 bucket name:

```bash
# Open the file
nano fix_g5_training.sh

# Change this line
S3_BUCKET="your-s3-bucket"  # Replace with your S3 bucket name
```

Ensure your SSH key is correctly specified:
```bash
SSH_KEY="$HOME/Downloads/gpu-key.pem"  # Verify this path is correct
```

## Step 4: Run the Fix Script

Execute the script to:
1. Stop the current training process
2. Download and extract the feature archives
3. Verify normalization stats
4. Restart training with real data

```bash
# Make the script executable
chmod +x fix_g5_training.sh

# Run the script
./fix_g5_training.sh
```

## Step 5: Monitor the Training

After applying the fix, you should see:

1. GPU utilization jump to 60-90%
2. Real data being processed (1440 RAVDESS files, 7440 CREMA-D files)
3. Normal training progress with proper epoch times

Use these commands to monitor:

```bash
# Check overall status and logs
./enhanced_monitor_g5.sh

# Set up TensorBoard visualization 
./setup_tensorboard_tunnel.sh
# Then open http://localhost:6006 in your browser
```

## Troubleshooting

### Issue: Feature archives not downloading
- Check S3 permissions
- Verify AWS CLI is installed on EC2
- Check S3 bucket name and file paths

### Issue: Training not using real data after fix
- Verify the feature directories exist: `ssh -i gpu-key.pem ubuntu@18.208.166.91 "ls -la ~/emotion-recognition/"`
- Check directory sizes: `ssh -i gpu-key.pem ubuntu@18.208.166.91 "du -sh ~/emotion-recognition/ravdess_features_facenet/"`
- Confirm normalization stats: `ssh -i gpu-key.pem ubuntu@18.208.166.91 "ls -la ~/emotion-recognition/models/dynamic_padding_no_leakage/"`

### Issue: GPU utilization still low
- Look for error messages: `ssh -i gpu-key.pem ubuntu@18.208.166.91 "tail -n 100 ~/emotion-recognition/logs/train_laugh_*.log"`
- Consider restarting the instance: `aws ec2 reboot-instances --instance-ids <instance-id>`

## Expected Outcome

After completing these steps, the G5 instance should be training with:
- Full RAVDESS dataset (≈1.6 GB)
- Full CREMA-D dataset (≈1.0 GB)
- Proper laughter manifest
- High GPU utilization (60-90%)
- Normal epoch times (5-10 minutes per epoch)

The training should complete all 100 epochs in approximately 8-10 hours.
