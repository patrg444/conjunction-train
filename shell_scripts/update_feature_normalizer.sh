#!/usr/bin/env bash
# Script to update the feature_normalizer.py file on EC2 to fix the import error

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

echo "Uploading fixed feature_normalizer.py to EC2 instance..."
scp -i "$SSH_KEY" fixed_feature_normalizer.py "$SSH_HOST":/home/ubuntu/emotion-recognition/scripts/feature_normalizer.py

echo "Verifying file was uploaded correctly..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cat /home/ubuntu/emotion-recognition/scripts/feature_normalizer.py | grep load_normalization_stats"

echo "Done! Now you can restart the training using ./start_training_20250419_131546.sh"
