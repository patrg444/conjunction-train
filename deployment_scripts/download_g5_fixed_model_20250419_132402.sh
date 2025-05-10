#!/usr/bin/env bash
# Download model from GPU instance
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

echo "Checking if training is complete..."
TRAINING_STATUS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "grep -c 'Training completed' ~/train_output.log || echo 0")

if [[ $TRAINING_STATUS -eq 0 ]]; then
    echo "Training not yet complete. Viewing current status:"
    ssh -i "$SSH_KEY" "$SSH_HOST" "tail -n 30 ~/train_output.log"
    exit 1
fi

echo "Downloading model from GPU instance..."
MODEL_DIR="models/g5_fixed_20250419_132402"
mkdir -p $MODEL_DIR

# Copy model files
scp -i "$SSH_KEY" "$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/model_best.h5" "$MODEL_DIR/"
scp -i "$SSH_KEY" "$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/model_info.json" "$MODEL_DIR/"
scp -i "$SSH_KEY" "$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/audio_normalization_stats.pkl" "$MODEL_DIR/"
scp -i "$SSH_KEY" "$SSH_HOST:~/emotion-recognition/models/dynamic_padding_no_leakage/video_normalization_stats.pkl" "$MODEL_DIR/"
scp -i "$SSH_KEY" "$SSH_HOST:~/train_output.log" "$MODEL_DIR/training.log"

echo "Model downloaded to $MODEL_DIR"
