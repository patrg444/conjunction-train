#!/bin/bash

# Basic evaluation script for R3D-18 model

# --- Configuration ---
# Provide the path to the checkpoint and test manifest as arguments
CHECKPOINT_PATH="$1" # e.g., training_logs/r3d18_experiment/checkpoints/r3d18-epoch=19-val_acc=0.85.ckpt
TEST_MANIFEST="$2"   # e.g., datasets/manifests/test_manifest.csv
OUTPUT_DIR="evaluation_results/$(basename ${CHECKPOINT_PATH%.*})_eval_$(date +%Y%m%d_%H%M%S)" # Unique output dir
BATCH_SIZE=16 # Adjust based on GPU memory
NUM_WORKERS=4
SAVE_CM_PLOT=true # Set to true to save confusion matrix plot

# --- Input Validation ---
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path argument is required."
    echo "Usage: $0 <path_to_checkpoint.ckpt> <path_to_test_manifest.csv>"
    exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at ${CHECKPOINT_PATH}"
    exit 1
fi
if [ -z "$TEST_MANIFEST" ]; then
    echo "Error: Test manifest path argument is required."
    echo "Usage: $0 <path_to_checkpoint.ckpt> <path_to_test_manifest.csv>"
    exit 1
fi
if [ ! -f "$TEST_MANIFEST" ]; then
    echo "Error: Test manifest file not found at ${TEST_MANIFEST}"
    exit 1
fi


# --- Activate Environment (if necessary) ---
# source /path/to/your/venv/bin/activate # Example: Uncomment and modify if using a virtual environment

# --- Run Evaluation ---
echo "Starting R3D-18 evaluation..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Test Manifest: ${TEST_MANIFEST}"
echo "Output Directory: ${OUTPUT_DIR}"

CMD="python eval_r3d18.py \
    --checkpoint_path \"${CHECKPOINT_PATH}\" \
    --test_manifest \"${TEST_MANIFEST}\" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --output_dir \"${OUTPUT_DIR}\""

if [ "$SAVE_CM_PLOT" = true ]; then
    CMD="$CMD --save_cm_plot"
fi

echo "Running command: $CMD"
eval $CMD


# --- Check Exit Status ---
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "Results saved in ${OUTPUT_DIR}"
else
    echo "Evaluation failed."
    exit 1
fi

echo "Evaluation script finished."
