#!/bin/bash
set -e

# Add the current directory to PYTHONPATH to make the imports work
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Train the Text-Humor branch model with proper manifest paths for the train and validation sets
python train_distil_humor.py \
  --train_manifest datasets/manifests/humor/train_humor_with_text.csv \
  --val_manifest datasets/manifests/humor/val_humor_with_text.csv \
  --model_name distilbert-base-uncased \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --log_dir logs/text_humor \
  --exp_name distilbert_humor_training

# Copy the best checkpoint to the expected location for downstream tasks
mkdir -p checkpoints
BEST_CHECKPOINT=$(find logs/text_humor/distilbert_humor_training/lightning_logs/version_*/checkpoints/ -name "*.ckpt" | sort -r | head -n 1)

if [ -n "$BEST_CHECKPOINT" ]; then
  echo "Found best checkpoint: $BEST_CHECKPOINT"
  cp "$BEST_CHECKPOINT" checkpoints/text_best.ckpt
  echo "Text-Humor training completed. Best checkpoint copied to: checkpoints/text_best.ckpt"
else
  echo "Warning: No checkpoint found!"
fi
