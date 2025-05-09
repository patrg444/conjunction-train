#!/bin/bash
# Download the final fixed wav2vec model (v7)

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_FILE="checkpoints/wav2vec_six_classes_best.weights.h5"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_MODEL="wav2vec_final_fixed_model_v7_$TIMESTAMP.h5"

echo "Downloading final fixed wav2vec model v7..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/$MODEL_FILE $LOCAL_MODEL

if [ -f "$LOCAL_MODEL" ]; then
    echo "Model successfully downloaded to $LOCAL_MODEL"
else
    echo "Error: Model download failed"
fi
