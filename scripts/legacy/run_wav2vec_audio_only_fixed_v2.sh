#!/bin/bash
# Run the fixed v2 version of the wav2vec audio emotion recognition training with improved numerical stability
# This version fixes the optimizer configuration to avoid the clipnorm/clipvalue conflict

# Set timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="wav2vec_fixed_training_v2_${TIMESTAMP}.log"

# Set the data directory - adjust if your data is in a different location
DATA_DIR="/home/ubuntu/audio_emotion/wav2vec_features"

# Configuration parameters
BATCH_SIZE=32
EPOCHS=150
LR=0.001
DROPOUT=0.5
MODEL_NAME="wav2vec_audio_only_fixed_v2_${TIMESTAMP}"

echo "Starting training with fixed wav2vec model (v2) at $(date)"
echo "Logs will be saved to: $LOG_FILE"

# Run the training script with parameters
python -m scripts.train_wav2vec_audio_only_fixed_v2 \
    --data_dir $DATA_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --dropout $DROPOUT \
    --model_name $MODEL_NAME \
    --debug \
    2>&1 | tee $LOG_FILE

echo "Training completed at $(date)"
echo "Check TensorBoard for training metrics:"
echo "tensorboard --logdir=logs"
