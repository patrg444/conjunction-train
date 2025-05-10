#!/bin/bash
set -e

echo "Starting Text-Only Fusion Humor training (CPU mode)..."

# Run the fixed train_fusion_humor_text_only script with CPU-friendly parameters
python scripts/fixed_train_fusion_humor_text_only.py \
  --config configs/train_humor.yaml \
  --epochs 10 \
  --batch_size 4 \
  --lr 0.0001

echo "Text-Only Fusion Humor training completed!"
