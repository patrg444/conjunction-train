#!/bin/bash
# Download the trained Wav2Vec emotion recognition model with fixed learning rate callback
# This script downloads the best weights from the server after training

SSH_KEY="~/Downloads/gpu-key.pem"
REMOTE_HOST="ubuntu@54.162.134.77"
REMOTE_WEIGHTS="/home/ubuntu/audio_emotion/checkpoints/wav2vec_audio_only_fixed_v4_restarted_20250422_132858_best.weights.h5"
LOCAL_DEST="./checkpoints/"

echo "Downloading Wav2Vec emotion recognition model weights..."
echo "Creating local checkpoint directory if it doesn't exist..."
mkdir -p $LOCAL_DEST

echo "Downloading model weights from the server..."
scp -i $SSH_KEY $REMOTE_HOST:$REMOTE_WEIGHTS $LOCAL_DEST

echo "Download complete. Model weights saved to $LOCAL_DEST"
