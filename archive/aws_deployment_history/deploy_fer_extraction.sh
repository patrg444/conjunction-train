#!/bin/bash

# EC2 connection details (with the correct SSH key path)
AWS_IP="54.162.134.77"
SSH_USER="ubuntu"
SSH_HOST="$SSH_USER@$AWS_IP"
SSH_KEY="$HOME/Downloads/gpu-key.pem"

# Check if the key exists
if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found at $SSH_KEY"
  exit 1
fi

# Ensure key has proper permissions
chmod 400 "$SSH_KEY" 2>/dev/null || true

# SSH command
SSH_COMMAND="ssh -i $SSH_KEY $SSH_HOST"
SCP_COMMAND="scp -i $SSH_KEY"

echo "Copying updated extract_fer_features.py to EC2..."
$SCP_COMMAND scripts/updated_extract_fer_features.py ${SSH_HOST}:/home/ubuntu/emotion-recognition/scripts/extract_fer_features.py

echo "Making the script executable..."
$SSH_COMMAND "chmod +x /home/ubuntu/emotion-recognition/scripts/extract_fer_features.py"

echo "Executing the FER feature extraction script..."
$SSH_COMMAND "cd /home/ubuntu/emotion-recognition && python scripts/extract_fer_features.py --dataset ravdess"

echo "Done! FER features have been extracted to the specified output directories."
echo "RAVDESS_VIDEOS_PATH=/home/ubuntu/datasets/ravdess_videos"
echo "CREMA_D_VIDEOS_PATH=/home/ubuntu/datasets/crema_d_videos"
