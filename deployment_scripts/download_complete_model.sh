#!/bin/bash
# Script to download the model trained with the complete solution

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_CHECKPOINT_DIR="/home/ubuntu/audio_emotion/checkpoints"
LOCAL_CHECKPOINT_DIR="./checkpoints_wav2vec_complete"

echo "===== Downloading Complete Wav2Vec Model ====="

# Create local directory
mkdir -p $LOCAL_CHECKPOINT_DIR

# Download the best model
echo "Downloading best model..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/best_model.h5 $LOCAL_CHECKPOINT_DIR/

# Download the final model
echo "Downloading final model..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/final_model.h5 $LOCAL_CHECKPOINT_DIR/

# Download label classes
echo "Downloading label encoder classes..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/label_classes.npy $LOCAL_CHECKPOINT_DIR/

# Download normalization parameters
echo "Downloading normalization parameters..."
scp -i $KEY_PATH $SERVER:/home/ubuntu/audio_emotion/audio_mean.npy $LOCAL_CHECKPOINT_DIR/
scp -i $KEY_PATH $SERVER:/home/ubuntu/audio_emotion/audio_std.npy $LOCAL_CHECKPOINT_DIR/

echo "===== Download Complete ====="
echo "Models saved to $LOCAL_CHECKPOINT_DIR"
