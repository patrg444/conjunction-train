#!/bin/bash
set -e

# Train the Text-Humor branch model with cleaned manifests
python train_distil_humor.py \
    --train_manifest datasets/manifests/humor/ur_funny_train_humor_cleaned.csv \
    --val_manifest datasets/manifests/humor/ur_funny_val_humor_cleaned.csv \
    --model_name "distilbert-base-uncased" \
    --max_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --epochs 5 \
    --num_workers 4 \
    --exp_name "ur_funny_cleaned"

echo "Text-Humor training completed. Best checkpoint at: checkpoints/text_best.ckpt"
