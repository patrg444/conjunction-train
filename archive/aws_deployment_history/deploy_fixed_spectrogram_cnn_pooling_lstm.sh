#!/bin/bash

# Script to deploy the fixed train_spectrogram_cnn_pooling_lstm.py to EC2

# Set variables
EC2_KEY="$HOME/Downloads/gpu-key.pem"
EC2_SERVER="ubuntu@54.162.134.77"
REMOTE_PATH="/home/ubuntu/emotion_project"
LOCAL_SCRIPT="fixed_train_spectrogram_cnn_pooling_lstm.py"
REMOTE_SCRIPT="train_spectrogram_cnn_pooling_lstm.py"
BACKUP_SCRIPT="train_spectrogram_cnn_pooling_lstm.py.bak"

# Display what we're doing
echo "Deploying fixed script to EC2..."

# Create backup of original script on server
echo "Creating backup of original script..."
ssh -i "$EC2_KEY" "$EC2_SERVER" "cd $REMOTE_PATH && cp $REMOTE_SCRIPT $BACKUP_SCRIPT"

# Upload fixed script to server
echo "Uploading fixed script to server..."
scp -i "$EC2_KEY" "$LOCAL_SCRIPT" "$EC2_SERVER:$REMOTE_PATH/$REMOTE_SCRIPT"

# Check if script passes syntax check
echo "Verifying script syntax..."
ssh -i "$EC2_KEY" "$EC2_SERVER" "cd $REMOTE_PATH && python3 -c 'import py_compile; py_compile.compile(\"$REMOTE_SCRIPT\", doraise=True)'" && echo "Syntax check passed!" || echo "Syntax check failed!"

echo "Deployment complete!"
echo "To run the script on EC2, use: ssh -i $EC2_KEY $EC2_SERVER \"cd $REMOTE_PATH && python3 $REMOTE_SCRIPT\""
