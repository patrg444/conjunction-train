#!/bin/bash
# Script to deploy and run audio-only CNN LSTM v2 training

# Set variables
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
SERVER="ubuntu@18.208.166.91"
LOCAL_SCRIPT="scripts/train_audio_only_cnn_lstm_v2.py"
SERVER_SCRIPT="/home/ubuntu/emotion-recognition/scripts/train_audio_only_cnn_lstm_v2.py"

# Copy the modified script to the server
echo "Copying training script to server..."
scp -i $SSH_KEY $LOCAL_SCRIPT $SERVER:$SERVER_SCRIPT

# SSH to the server and start training
echo "Starting training on server..."
ssh -i $SSH_KEY $SERVER "cd /home/ubuntu/emotion-recognition && \
  mkdir -p models && \
  echo 'Starting training at $(date)' && \
  nohup python3 $SERVER_SCRIPT > audio_only_training.log 2>&1 &"

echo "Training started on server. To monitor progress, use:"
echo "ssh -i $SSH_KEY $SERVER 'tail -f /home/ubuntu/emotion-recognition/audio_only_training.log'"
