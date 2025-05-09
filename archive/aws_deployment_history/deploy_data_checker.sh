#!/bin/bash
# Deploy and run the data checker script on EC2 to diagnose the data directory issue

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_SCRIPT="./scripts/check_wav2vec_data_directory.py"

echo "Deploying data checker script to EC2..."
scp -i $KEY_PATH $LOCAL_SCRIPT $EC2_HOST:$REMOTE_DIR/

echo "Running data checker script on various possible data directories..."

echo -e "\n=== Checking default directory ==="
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && python3 check_wav2vec_data_directory.py"

echo -e "\n=== Checking wav2vec_features directory ==="
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && python3 check_wav2vec_data_directory.py --data_dir=$REMOTE_DIR/wav2vec_features"

echo -e "\n=== Checking data directory ==="
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && python3 check_wav2vec_data_directory.py --data_dir=$REMOTE_DIR/data"

echo -e "\n=== Checking audio_emotion directory ==="
ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && python3 check_wav2vec_data_directory.py --data_dir=$REMOTE_DIR"

echo -e "\n=== Checking for wav2vec feature files recursively ==="
ssh -i $KEY_PATH $EC2_HOST "find $REMOTE_DIR -name '*.npz' | grep -v cache | head -20"

echo "Data directory check complete!"
