#!/bin/bash
set -e

echo "Starting Fusion Humor training..."

# Check if CelebA manifests exist, if not prepare them
if [ ! -f "datasets/manifests/humor/train_smile.csv" ] || [ ! -f "datasets/manifests/humor/val_smile.csv" ]; then
  echo "CelebA Smile manifests not found, preparing them..."
  python datasets/scripts/prepare_celeba_smile_manifest.py --celeba_root /home/ubuntu/datasets/celeba
fi

# Run the fixed train_fusion_humor script
python scripts/fixed_train_fusion_humor.py \
  --config configs/train_humor.yaml \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001 \
  --fp16 \
  --use_video \
  --use_text

echo "Fusion Humor training completed!"
