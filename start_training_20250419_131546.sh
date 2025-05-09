#!/usr/bin/env bash
# Script to start training on AWS g5.2xlarge instance

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

EPOCHS=100
BATCH_SIZE=256 
MAX_SEQ_LEN=45  # Keep MAX_SEQ_LEN=45 to stay compatible with realtime window
LAUGH_WEIGHT=0.3

echo "Starting training on EC2 instance with $EPOCHS epochs, batch size $BATCH_SIZE..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cd ~/emotion-recognition && chmod +x train_g5_20250419_131350.sh && ./train_g5_20250419_131350.sh"

echo "Training job has been started on the EC2 instance."
echo "Use ./monitor_training_20250419_131515.sh to monitor training progress."
echo "Use ./download_g5_model_20250419_131350.sh to download the model after training is complete."
