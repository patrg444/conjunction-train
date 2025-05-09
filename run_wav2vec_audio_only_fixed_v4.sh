#!/bin/bash
# Script to run the fixed wav2vec audio emotion recognition model (v4)
# This version fixes all syntax errors, improves numerical stability,
# and handles both NPZ and NPY wav2vec feature files

# Set the data directory - adjust as needed based on your EC2 setup
# Corrected path based on previous 'find' command results
DATA_DIR="/home/ubuntu/audio_emotion/models/wav2vec"

# Ensure checkpoints and logs directories exist
mkdir -p checkpoints
mkdir -p logs

# Log file to capture output - Use the correct v4 log name
LOG_FILE="train_wav2vec_audio_only_fixed_v4.log"

# Run the training script with optimized parameters
echo "Starting wav2vec audio emotion training with fixed v4 script..."
echo "Log file: $LOG_FILE"

# Training command with early stopping
python train_wav2vec_audio_only_fixed_v4.py \
  --data_dir $DATA_DIR \
  --batch_size 64 \
  --epochs 100 \
  --lr 0.001 \
  --dropout 0.5 \
  --use_cache # Removed tee command as nohup handles redirection

echo "Training completed. Check $LOG_FILE for details."
