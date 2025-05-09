#!/bin/bash
# Script to run training on AWS with GPU acceleration

# Default parameters
LEARNING_RATE="1e-5"
EPOCHS=50
BATCH_SIZE=32
USE_S3="false"
S3_BUCKET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --use-s3)
      USE_S3="true"
      S3_BUCKET="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Detect number of available GPUs and configure environment for GPU training
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Check if multiple GPUs are available
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Detected $NUM_GPUS GPUs. Enabling multi-GPU training."
  # Use all available GPUs
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  # Enable memory optimization for high RAM instances
  export TF_GPU_ALLOCATOR=cuda_malloc_async
  # Increase batch size if we have multiple GPUs and sufficient RAM
  if [ "$BATCH_SIZE" -lt 64 ]; then
    echo "Increasing batch size from $BATCH_SIZE to $((BATCH_SIZE * 2)) for multi-GPU training."
    BATCH_SIZE=$((BATCH_SIZE * 2))
  fi
else
  echo "Single GPU detected. Running standard GPU training."
  export CUDA_VISIBLE_DEVICES=0
fi

# Configure memory settings for high RAM instances
export TF_MEMORY_ALLOCATION=0.85  # Use up to 85% of available RAM

echo "Starting training with GPU acceleration..."
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

# Create a timestamp for this training run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="training_${TIMESTAMP}.log"

# Run training with specified parameters
python train_branched_6class.py \
  --ravdess-dir ravdess_features_facenet \
  --cremad-dir crema_d_features_facenet \
  --learning-rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --model-dir models/branched_6class \
  --eval-dir models/evaluation_${TIMESTAMP} | tee $LOGFILE

# Upload results to S3 if specified
if [ "$USE_S3" = "true" ] && [ -n "$S3_BUCKET" ]; then
  echo "Uploading models and logs to S3 bucket: $S3_BUCKET"
  aws s3 sync models/branched_6class s3://$S3_BUCKET/models/branched_6class
  aws s3 sync models/evaluation_${TIMESTAMP} s3://$S3_BUCKET/evaluation/${TIMESTAMP}
  aws s3 cp $LOGFILE s3://$S3_BUCKET/logs/${LOGFILE}
  echo "Upload complete."
fi

echo "Training complete. Check $LOGFILE for details."
