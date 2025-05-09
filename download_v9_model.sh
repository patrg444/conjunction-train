#!/bin/bash
# Script to download the V9 attention model trained on WAV2VEC features

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77"
REMOTE_CHECKPOINT_DIR="/home/ubuntu/audio_emotion/checkpoints"
LOCAL_CHECKPOINT_DIR="./checkpoints_wav2vec_v9_attention"

echo "===== Downloading WAV2VEC V9 Attention Model ====="

# Create local directory
mkdir -p $LOCAL_CHECKPOINT_DIR

# Download the best model
echo "Downloading best model..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/best_model_v9.h5 $LOCAL_CHECKPOINT_DIR/

# Download the final model
echo "Downloading final model..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/final_model_v9.h5 $LOCAL_CHECKPOINT_DIR/

# Download label classes
echo "Downloading label encoder classes..."
scp -i $KEY_PATH $SERVER:$REMOTE_CHECKPOINT_DIR/label_classes_v9.npy $LOCAL_CHECKPOINT_DIR/

# Download normalization parameters
echo "Downloading normalization parameters..."
scp -i $KEY_PATH $SERVER:/home/ubuntu/audio_emotion/audio_mean_v9.npy $LOCAL_CHECKPOINT_DIR/
scp -i $KEY_PATH $SERVER:/home/ubuntu/audio_emotion/audio_std_v9.npy $LOCAL_CHECKPOINT_DIR/

echo "===== Download Complete ====="
echo "Models saved to $LOCAL_CHECKPOINT_DIR"
