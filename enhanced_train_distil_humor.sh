#!/bin/bash
set -e

echo "Starting enhanced DistilBERT Humor classifier training with detailed logging..."

# Set up directories if they don't exist
mkdir -p checkpoints
mkdir -p training_logs_text_humor

# Run the enhanced training script with improved logging
python enhanced_train_distil_humor.py \
  --train_manifest datasets/manifests/humor/train_humor_with_text.csv \
  --val_manifest datasets/manifests/humor/val_humor_with_text.csv \
  --model_name distilbert-base-uncased \
  --max_length 128 \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --num_workers 4 \
  --log_dir training_logs_text_humor

echo "Training completed. Check training_logs_text_humor for detailed metrics and checkpoints/text_best.ckpt for the best model."
