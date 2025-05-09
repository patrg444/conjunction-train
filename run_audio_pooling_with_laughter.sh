#!/usr/bin/env bash
# Script to run audio-pooling LSTM training with laughter detection
# Usage: bash run_audio_pooling_with_laughter.sh [epochs] [batch_size] [max_seq_len] [laugh_weight]
#
# Default values:
# - epochs: 100
# - batch_size: 32 for local, 256 for GPU
# - max_seq_len: 45 for compatibility with real-time window
# - laugh_weight: 0.3 for auxiliary laughter loss weight

# Get arguments
EPOCHS=${1:-100}
BATCH_SIZE=${2:-32}
MAX_SEQ_LEN=${3:-45}
LAUGH_WEIGHT=${4:-0.3}

# Check if running on a GPU instance
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using larger batch size"
    BATCH_SIZE=${2:-256}
fi

# Set output directory
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
MODEL_DIR="models/audio_pooling_with_laughter_${TIMESTAMP}"
mkdir -p ${MODEL_DIR}
mkdir -p logs

# Start training
echo "Starting training with parameters:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max sequence length: ${MAX_SEQ_LEN}"
echo "  Laughter loss weight: ${LAUGH_WEIGHT}"
echo "  Model directory: ${MODEL_DIR}"

# Check if laughter manifest exists
if [ -f "datasets_raw/manifests/laughter_v1.csv" ]; then
    echo "Laughter manifest found, training with laughter detection"
else
    echo "Warning: Laughter manifest not found, setup data first with 'make laughter_data'"
    echo "         Will train without laughter detection"
fi

# Run training
python3 scripts/train_audio_pooling_lstm.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --model_dir ${MODEL_DIR} \
    2>&1 | tee logs/train_${TIMESTAMP}.log

echo "Training completed. Model saved to ${MODEL_DIR}"

# Create monitoring script
cat > monitor_audio_pooling_with_laughter_${TIMESTAMP}.sh <<EOF
#!/usr/bin/env bash
# Script to monitor training progress
watch -n 10 "grep 'loss|val_loss|accuracy|val_accuracy' logs/train_laugh_${TIMESTAMP}.log | tail -20"
EOF

chmod +x monitor_audio_pooling_with_laughter_${TIMESTAMP}.sh
echo "Created monitoring script: monitor_audio_pooling_with_laughter_${TIMESTAMP}.sh"

# Create model info json
cat > ${MODEL_DIR}/model_info.json <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "model_type": "audio_pooling_lstm_with_laughter",
  "epochs": ${EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "max_seq_len": ${MAX_SEQ_LEN},
  "laugh_weight": ${LAUGH_WEIGHT},
  "architecture": "LSTM with auxiliary laughter detection branch"
}
EOF

echo "Model info saved to ${MODEL_DIR}/model_info.json"
