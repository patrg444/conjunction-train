#!/bin/bash
set -e

CELEBA_ROOT="/home/ubuntu/datasets/celeba"
MANIFEST_DIR="datasets/manifests/humor"
TRAIN_MANIFEST="$MANIFEST_DIR/train_smile.csv"
VAL_MANIFEST="$MANIFEST_DIR/val_smile.csv"

# Create the manifest directory if it doesn't exist
mkdir -p $MANIFEST_DIR

# Check if the manifests exist, if not generate them
if [ ! -f "$TRAIN_MANIFEST" ] || [ ! -f "$VAL_MANIFEST" ]; then
    echo "Manifest files not found. Preparing CelebA smile manifests..."
    python datasets/scripts/prepare_celeba_smile_manifest.py --celeba_root $CELEBA_ROOT
fi

# Train the model
python train_smile.py --train_manifest $TRAIN_MANIFEST --val_manifest $VAL_MANIFEST

echo "Smile training completed. Best checkpoint at: checkpoints/smile_best.ckpt"
