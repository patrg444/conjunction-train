#!/bin/bash
# Download just the emotion labels file from the WAV2VEC model on EC2

echo "===== Downloading Emotion Labels for Model Verification ====="

# Set the EC2 connection details
EC2_USER="ubuntu"
EC2_HOST=$(cat aws_instance_ip.txt 2>/dev/null)

if [ -z "$EC2_HOST" ]; then
  echo "Error: EC2 host IP not found. Please check aws_instance_ip.txt file."
  exit 1
fi

# Create directory for model files if it doesn't exist
mkdir -p models/wav2vec_v9_attention

# Download just the label classes file
echo "Downloading emotion label mappings..."
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/checkpoints/label_classes_v9.npy models/wav2vec_v9_attention/

# Check if file was downloaded successfully
if [ -f "models/wav2vec_v9_attention/label_classes_v9.npy" ]; then
  echo "Emotion labels downloaded successfully."
  echo "To verify the labels, run: ./verify_emotion_labels.py"
else
  echo "Error: Failed to download emotion labels."
  exit 1
fi

echo "===== Download Complete ====="
