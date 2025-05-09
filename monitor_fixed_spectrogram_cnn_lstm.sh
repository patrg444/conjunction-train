#!/bin/bash

# Script to run a short test of the fixed train_spectrogram_cnn_pooling_lstm.py on EC2

# Set variables
EC2_KEY="$HOME/Downloads/gpu-key.pem"
EC2_SERVER="ubuntu@54.162.134.77"
REMOTE_PATH="/home/ubuntu/emotion_project"
REMOTE_SCRIPT="train_spectrogram_cnn_pooling_lstm.py"

# Display what we're doing
echo "Starting a test run of the fixed script on EC2..."
echo "This will start the training process and show the first minute of output to verify it's working"

# Run the script and monitor the output for 60 seconds
ssh -i "$EC2_KEY" "$EC2_SERVER" "cd $REMOTE_PATH && timeout 60s python3 $REMOTE_SCRIPT"

echo ""
echo "Test run complete. The script should have started successfully if no syntax errors were shown."
echo "To run the full training, use: ssh -i $EC2_KEY $EC2_SERVER \"cd $REMOTE_PATH && python3 $REMOTE_SCRIPT\""
