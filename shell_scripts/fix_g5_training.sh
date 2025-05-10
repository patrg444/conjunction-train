#!/usr/bin/env bash
# Script to fix the G5 training issue by uploading real data

# Set constants
SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"
S3_BUCKET="emotion-recognition-data"  # S3 bucket with feature archives

# Check SSH connection
echo "Testing SSH connection..."
ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" "echo Connected successfully" || {
    echo "SSH connection failed. Please check your SSH key and security settings."
    exit 1
}

# Stop the current training process that's using dummy data
echo "Stopping current training process..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    pgrep -f 'train_audio_pooling_lstm_with_laughter.py' | xargs --no-run-if-empty kill && \
    echo 'Stopped existing training process'"

# Create directories if they don't exist
echo "Creating directories on EC2 instance..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    mkdir -p ravdess_features_facenet && \
    mkdir -p crema_d_features_facenet && \
    mkdir -p datasets/manifests && \
    mkdir -p models/dynamic_padding_no_leakage"

# 1. Upload and extract feature archives (using S3)
echo "Uploading feature archives to S3..."
# Note: Placeholder - requires actual archive files
# aws s3 cp ravdess_features_facenet.tar.gz s3://$S3_BUCKET/
# aws s3 cp crema_d_features_facenet.tar.gz s3://$S3_BUCKET/

echo "Downloading and extracting feature archives on EC2..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    aws s3 cp s3://$S3_BUCKET/ravdess_features_facenet.tar.gz . && \
    aws s3 cp s3://$S3_BUCKET/crema_d_features_facenet.tar.gz . && \
    echo 'Starting extraction of RAVDESS features...' && \
    tar xzf ravdess_features_facenet.tar.gz && \
    echo 'Starting extraction of CREMA-D features...' && \
    tar xzf crema_d_features_facenet.tar.gz && \
    echo 'Extraction complete'"

# 2. Upload proper laughter manifest
echo "Creating proper laughter manifest..."
# Note: Placeholder - requires actual manifest file
# If you have the actual laughter_v1.csv:
# scp -i "$SSH_KEY" datasets/manifests/laughter_v1.csv "$SSH_HOST":~/emotion-recognition/datasets/manifests/

# 3. Ensure normalization stats are present
echo "Checking normalization statistics files..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    ls -la models/dynamic_padding_no_leakage/*_normalization_stats.pkl || \
    echo 'Normalization files missing'"

# Upload normalization files if needed
# scp -i "$SSH_KEY" audio_normalization_stats.pkl "$SSH_HOST":~/emotion-recognition/models/dynamic_padding_no_leakage/
# scp -i "$SSH_KEY" video_normalization_stats.pkl "$SSH_HOST":~/emotion-recognition/models/dynamic_padding_no_leakage/

# 4. Verify data presence and size
echo "Verifying uploaded data..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    echo 'RAVDESS features:' && du -sh ravdess_features_facenet/ && \
    echo 'CREMA-D features:' && du -sh crema_d_features_facenet/ && \
    echo 'Normalization files:' && find models -name '*normalization_stats.pkl' | xargs ls -lh"

# 5. Restart training with real data
echo "Restarting training process..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && \
    nohup bash run_audio_pooling_with_laughter.sh 100 256 45 0.3 > logs/training_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"

# 6. Verify GPU utilization has increased
echo "Monitoring GPU utilization (should increase to 60-90%)..."
ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"

echo "Setup complete! To monitor training progress:"
echo "1. Run './enhanced_monitor_g5.sh' to see logs and GPU usage"
echo "2. Run './setup_tensorboard_tunnel.sh' to set up visualization"
echo ""
echo "IMPORTANT: This script assumes you have uploaded the feature archives to S3."
echo "If not, modify the script to upload directly or use another transfer method."
