#!/bin/bash
# Script to train a humor classifier on combined datasets

python enhanced_train_distil_humor.py --train_manifest datasets/manifests/humor/combined_train_humor.csv --val_manifest datasets/manifests/humor/combined_val_humor.csv --batch_size 32 --epochs 5 --learning_rate 5e-5 --model_name 'distilbert-base-uncased' --output_dir './checkpoints/humor_classifier/' --max_length 128
