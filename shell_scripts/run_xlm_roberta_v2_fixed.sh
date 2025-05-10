#!/bin/bash
# XLM-RoBERTa v2 Training Launcher with Optimized Configuration

# Fixed paths to point to full UR-Funny dataset on EC2
# Using the same paths as the DeBERTa v3 model
TRAIN_MANIFEST="datasets/manifests/humor/ur_funny_train_humor_cleaned.csv"
VAL_MANIFEST="datasets/manifests/humor/ur_funny_val_humor_cleaned.csv"

# Model configuration
MODEL_NAME="xlm-roberta-large"
MAX_LENGTH=128
BATCH_SIZE=8   # Increase if you have more GPU memory
LR=2e-5        # Increased from 1e-5 for faster convergence
EPOCHS=15
NUM_WORKERS=4
WEIGHT_DECAY=0.01
DROPOUT=0.1
SCHEDULER="cosine"  # cosine scheduler works better with longer training
GRAD_CLIP=1.0

# Experiment configuration
LOG_DIR="training_logs_humor"
EXP_NAME="xlm-roberta-large_optimized"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}/${EXP_NAME}/checkpoints"

# Check for GPU and set parameters accordingly
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling mixed precision training"
    FP16="--fp16"
    DEVICES="--devices 1"
else
    echo "No GPU detected, using CPU only"
    FP16=""
    DEVICES="--devices 1"  # Use 1 CPU device instead of 0, which is invalid
fi

# Print configuration
echo "=== XLM-RoBERTa v2 Training Configuration ==="
echo "Training manifest: $TRAIN_MANIFEST"
echo "Validation manifest: $VAL_MANIFEST"
echo "Model: $MODEL_NAME"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Scheduler: $SCHEDULER"
echo "FP16: ${FP16:-Disabled}"
echo "=== Additional Features ==="
echo "- Dynamic padding (memory efficient)"
echo "- Class weight balancing"
echo "- Monitored metric: val_f1"
echo "- Corrected scheduler steps"
echo "- Reproducible with fixed seed"
echo "- Auto-detection of hardware"

# Run the training script
python fixed_train_xlm_roberta_script_v2.py \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --model_name "$MODEL_NAME" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --epochs "$EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --scheduler "$SCHEDULER" \
    --grad_clip "$GRAD_CLIP" \
    --log_dir "$LOG_DIR" \
    --exp_name "$EXP_NAME" \
    --class_balancing \
    --monitor_metric "val_f1" \
    --seed 42 \
    $DEVICES $FP16

# After training completes, start the monitoring script
echo "Training started! Launching monitoring script..."
./monitor_xlm_roberta_v2_training.sh
