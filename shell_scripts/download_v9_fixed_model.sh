#!/bin/bash
# Download the WAV2VEC v9 attention-based emotion recognition model from EC2

echo "===== Downloading Fixed WAV2VEC Attention Model (v9) ====="

# Set the EC2 connection details
EC2_USER="ubuntu"
EC2_HOST=$(cat aws_instance_ip.txt 2>/dev/null)

if [ -z "$EC2_HOST" ]; then
  echo "Error: EC2 host IP not found. Please check aws_instance_ip.txt file."
  exit 1
fi

# Create directory for model files
mkdir -p models/wav2vec_v9_attention

# Download the model files
echo "Downloading model files..."
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/checkpoints/best_model_v9.h5 models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/checkpoints/label_classes_v9.npy models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/audio_mean_v9.npy models/wav2vec_v9_attention/
scp -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST:/home/ubuntu/audio_emotion/audio_std_v9.npy models/wav2vec_v9_attention/

# Check if files were downloaded successfully
if [ -f "models/wav2vec_v9_attention/best_model_v9.h5" ]; then
  echo "Model downloaded successfully."
  echo "Files saved to models/wav2vec_v9_attention/"
  
  # Display model information
  echo ""
  echo "Model details:"
  echo "- Architecture: WAV2VEC with Attention mechanism"
  echo "- Validation accuracy: 85.00%"
  echo "- Training epochs: 30 (early stopping at epoch 18)"
  echo "- Features: WAV2VEC embeddings"
else
  echo "Error: Failed to download model files."
  exit 1
fi

echo "===== Download Complete ====="
