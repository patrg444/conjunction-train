#!/bin/bash
# Script to convert wav2vec embeddings to expected format and run training

echo "Preparing wav2vec data and starting training..."

# Parameters
EC2_USER="ubuntu"
EC2_IP="54.162.134.77"
SSH_KEY="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Upload the data preparation script
echo "Uploading data preparation script to EC2..."
scp -i $SSH_KEY scripts/prepare_wav2vec_data.py $EC2_USER@$EC2_IP:$REMOTE_DIR/

# Stop any existing training processes
echo "Stopping any existing training processes..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "pkill -f train_wav2vec_lstm || true"

# Run the data preparation script
echo "Running data preparation script to convert .npz files to .npy and organize them..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && python prepare_wav2vec_data.py"

# Run the original training script with the prepared data
echo "Starting training with correctly prepared data..."
ssh -i $SSH_KEY $EC2_USER@$EC2_IP "cd $REMOTE_DIR && nohup python train_wav2vec_lstm.py --features_dir data --mean_path embedding_mean.npy --std_path embedding_std.npy --epochs 100 --batch_size 64 > train_wav2vec_lstm_with_prep.log 2>&1 &"

echo "Data preparation and training started!"
echo "Monitor training with:"
echo "ssh -i $SSH_KEY $EC2_USER@$EC2_IP \"tail -f $REMOTE_DIR/train_wav2vec_lstm_with_prep.log\""
