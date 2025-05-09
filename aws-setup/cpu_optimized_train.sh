#!/bin/bash
# CPU-optimized training script

# CPU optimization environment variables
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# For Intel CPUs (c5 instances use Intel CPUs)
export TF_ENABLE_ONEDNN_OPTS=1

# Disable CUDA to ensure CPU usage
export CUDA_VISIBLE_DEVICES=""

# Run the training
cd ~/emotion_training
python scripts/train_branched_6class.py \
  --ravdess-dir ravdess_features_facenet \
  --cremad-dir crema_d_features_facenet \
  --learning-rate 1e-4 \
  --epochs 50 \
  --batch-size 128 \
  --model-dir models/branched_6class \
  --use-class-weights
