#!/bin/bash
# Script to run audio-only wav2vec emotion recognition training on EC2

echo "Starting audio-only wav2vec emotion recognition training..."

# Parameters
EC2_USER="ubuntu"
EC2_IP="54.162.134.77"  # Update with your EC2 instance IP
SSH_KEY="~/Downloads/gpu-key.pem"  # Update with your SSH key path
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Create necessary directories
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "mkdir -p $REMOTE_DIR/scripts"

# Upload both the training script and data preparation script
echo "Uploading scripts to EC2..."
scp -i $SSH_KEY scripts/train_wav2vec_audio_only.py $EC2_USER@$EC2_IP:$REMOTE_DIR/scripts/
scp -i $SSH_KEY scripts/prepare_wav2vec_data.py $EC2_USER@$EC2_IP:$REMOTE_DIR/scripts/

# Ensure data directory structure is ready
echo "Running data preparation script..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && python scripts/prepare_wav2vec_data.py"

# Create directories for logs and checkpoints
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/checkpoints"

# Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "pkill -f train_wav2vec_audio_only || true"

# Run the audio-only training script with the prepared data
echo "Starting audio-only training..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && nohup python scripts/train_wav2vec_audio_only.py \
    --features_dir data \
    --mean_path embedding_mean.npy \
    --std_path embedding_std.npy \
    --epochs 100 \
    --batch_size 64 \
    --lr 6e-4 \
    --log_dir logs \
    --checkpoint_dir checkpoints \
    --seed 42 \
    > train_wav2vec_audio_only.log 2>&1 &"

echo "Audio-only training started!"
echo "Monitor training with:"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_IP \"tail -f $REMOTE_DIR/train_wav2vec_audio_only.log\""

# Instructions for downloading the model later
echo ""
echo "When training completes, download the model with:"
echo "scp -i $SSH_KEY $EC2_USER@$EC2_IP:\"$REMOTE_DIR/checkpoints/wav2vec_audio_only_*_best.h5\" ."
