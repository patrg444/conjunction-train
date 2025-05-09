#!/bin/bash
set -e

echo "Starting Text-Only Fusion Humor training..."

# Run the fixed train_fusion_humor_text_only script
python scripts/fixed_train_fusion_humor_text_only.py \
  --config configs/train_humor.yaml \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001 \
  --fp16

echo "Text-Only Fusion Humor training completed!"
