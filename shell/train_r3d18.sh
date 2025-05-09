#!/bin/bash

# Basic training script for R3D-18 model

# --- Configuration ---
TRAIN_MANIFEST="datasets/manifests/train_manifest.csv" # Replace with actual path
VAL_MANIFEST="datasets/manifests/val_manifest.csv"   # Replace with actual path
NUM_CLASSES=8 # Replace with actual number of classes for your dataset
EPOCHS=20
BATCH_SIZE=16 # Adjust based on GPU memory
LEARNING_RATE=1e-4
NUM_WORKERS=4
LOG_DIR="training_logs"
EXP_NAME="r3d18_$(date +%Y%m%d_%H%M%S)" # Unique experiment name with timestamp

# --- Activate Environment (if necessary) ---
# source /path/to/your/venv/bin/activate # Example: Uncomment and modify if using a virtual environment

# --- Run Training ---
echo "Starting R3D-18 training..."
echo "Experiment Name: ${EXP_NAME}"
echo "Log Directory: ${LOG_DIR}/${EXP_NAME}"

python train_r3d18.py \
    --train_manifest "${TRAIN_MANIFEST}" \
    --val_manifest "${VAL_MANIFEST}" \
    --num_classes ${NUM_CLASSES} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_workers ${NUM_WORKERS} \
    --log_dir "${LOG_DIR}" \
    --exp_name "${EXP_NAME}"

# --- Check Exit Status ---
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed."
    exit 1
fi

echo "Training script finished."
