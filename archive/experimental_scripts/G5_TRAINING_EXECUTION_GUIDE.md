# G5 Training Execution Guide

This guide provides step-by-step instructions for executing the G5 Training Fix toolkit. Follow these steps in order to resolve the training issues and properly restart the training with the real datasets.

## Prerequisites

- AWS CLI configured with access to the S3 bucket "emotion-recognition-data"
- SSH access to the G5 instance (key in `~/Downloads/gpu-key.pem`)
- The feature archives uploaded to S3 (or ready to be uploaded)

## Step 1: Verify EC2 Instance Status

```bash
# Check if the EC2 instance is running
aws ec2 describe-instances --filters "Name=ip-address,Values=18.208.166.91" --query "Reservations[*].Instances[*].State.Name" --output text

# Test SSH connection
ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "echo Connected successfully"
```

## Step 2: Generate the Laughter Manifest

The manifest has already been generated, but you can regenerate it if needed:

```bash
# Generate manifest with 500 samples
./generate_laughter_manifest.py --output datasets/manifests/laughter_v1.csv --samples 500
```

## Step 3: Check If Feature Archives Exist in S3

```bash
# Check if RAVDESS archive exists
aws s3 ls s3://emotion-recognition-data/ravdess_features_facenet.tar.gz

# Check if CREMA-D archive exists
aws s3 ls s3://emotion-recognition-data/crema_d_features_facenet.tar.gz
```

If the archives don't exist, you'll need to upload them:

```bash
# If you have the extracted feature directories locally, create archives
tar -czf ravdess_features_facenet.tar.gz ravdess_features_facenet/
tar -czf crema_d_features_facenet.tar.gz crema_d_features_facenet/

# Upload to S3
aws s3 cp ravdess_features_facenet.tar.gz s3://emotion-recognition-data/
aws s3 cp crema_d_features_facenet.tar.gz s3://emotion-recognition-data/
```

## Step 4: Run the One-Step Fix Script

```bash
# Execute the all-in-one fix script
./fix_and_restart_g5_training.sh
```

This script will:
1. Check SSH connection
2. Generate laughter manifest (if not already done)
3. Stop any existing training processes
4. Create necessary directories on EC2
5. Upload manifest and normalization fix script
6. Download and extract feature archives from S3
7. Fix normalization statistics if needed
8. Restart training with real data
9. Show initial GPU status

## Step 5: Monitor Training Progress

```bash
# Start continuous monitoring
./continuous_g5_monitor.sh

# Or for a shorter update interval (15 seconds)
./continuous_g5_monitor.sh 15
```

The monitoring script provides real-time information on:
- Training process status
- Dataset presence and sizes
- GPU utilization and memory
- Training progress with ETA
- Error detection

## Step 6: Set up TensorBoard (Optional)

For detailed training metrics visualization:

```bash
# Create SSH tunnel for TensorBoard
ssh -i ~/Downloads/gpu-key.pem -L 6006:localhost:6006 ubuntu@18.208.166.91
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

If the fix script doesn't complete successfully:

1. **SSH Connection Issues**
   ```bash
   # Check SSH key permissions
   chmod 400 ~/Downloads/gpu-key.pem
   ```

2. **S3 Access Issues**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Verify EC2 instance has proper IAM role with S3 access
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "aws s3 ls"
   ```

3. **Training Not Using Real Data**
   ```bash
   # Check directory sizes manually
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "du -sh ~/emotion-recognition/ravdess_features_facenet/ ~/emotion-recognition/crema_d_features_facenet/"
   
   # Verify manifest file
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "head -n 10 ~/emotion-recognition/datasets/manifests/laughter_v1.csv"
   ```

4. **GPU Utilization Still Low**
   ```bash
   # Check training logs for errors
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "tail -n 100 ~/emotion-recognition/logs/train_laugh_*.log"
   
   # Verify normalization files
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "ls -l ~/emotion-recognition/models/dynamic_padding_no_leakage/*_normalization_stats.pkl"
   ```

5. **Fix Normalization Stats Manually**
   ```bash
   # Run normalization fix script directly
   ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "cd ~/emotion-recognition && python fix_normalization_stats.py"
   ```

## Verification of Success

After applying the fixes, verify success with:

```bash
# Check GPU utilization (should be 60-90%)
ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"

# Check feature directories size (RAVDESS should be ~1.6GB, CREMA-D ~1.0GB)
ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "du -sh ~/emotion-recognition/ravdess_features_facenet/ ~/emotion-recognition/crema_d_features_facenet/"

# Check training progress
ssh -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91 "tail -n 20 ~/emotion-recognition/logs/train_laugh_*.log"
```

## Downloading Trained Models

Once training completes, download the model:

```bash
# Create a local directory for models
mkdir -p trained_models

# Download best model
scp -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91:~/emotion-recognition/models/audio_pooling_lstm_with_laughter_*.h5 trained_models/

# Download training logs
scp -i ~/Downloads/gpu-key.pem ubuntu@18.208.166.91:~/emotion-recognition/logs/train_laugh_*.log trained_models/
```

## Next Steps

After successful training, consider:

1. Evaluating the model on test data
2. Deploying the model for real-time inference
3. Running additional training with different parameters
4. Expanding the dataset with more laughter samples
