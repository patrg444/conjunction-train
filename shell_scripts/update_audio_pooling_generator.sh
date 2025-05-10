#!/usr/bin/env bash
# Script to update the audio_pooling_generator.py file on EC2 to fix the initialization parameters

SSH_KEY="$HOME/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

echo "Uploading fixed audio_pooling_generator.py to EC2 instance..."
scp -i "$SSH_KEY" fixed_audio_pooling_generator.py "$SSH_HOST":/home/ubuntu/emotion-recognition/scripts/audio_pooling_generator.py

echo "Verifying file was uploaded correctly..."
ssh -i "$SSH_KEY" "$SSH_HOST" "cat /home/ubuntu/emotion-recognition/scripts/audio_pooling_generator.py | grep -A 5 'def __init__'"

echo "Done! Now you can restart the training using ./start_training_20250419_131546.sh"
