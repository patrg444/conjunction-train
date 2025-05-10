#!/bin/bash
# Download the wav2vec fixed model (v4) from EC2
# This version supports models trained with the Keras-compatible attention approach

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
LOCAL_DIR="./downloaded_models"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create local directory if it doesn't exist
mkdir -p $LOCAL_DIR

echo "Fetching best checkpoint from EC2 instance..."

# First find the latest model checkpoint directory
LATEST_MODEL=$(ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR/checkpoints && ls -t wav2vec_audio_only_fixed_v4_* 2>/dev/null | head -1 | xargs basename || echo ''")

if [ -z "$LATEST_MODEL" ]; then
  echo "No wav2vec_audio_only_fixed_v4 model found on the server."
  echo "Make sure training has completed and checkpoint files were generated."
  exit 1
fi

MODEL_BASE=${LATEST_MODEL%_*}  # Remove _best.weights.h5 suffix if present

echo "Found model: $MODEL_BASE"
echo "Downloading model files..."

# Download model files with scp
echo "Downloading weights file..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/checkpoints/${MODEL_BASE}_best.weights.h5 $LOCAL_DIR/

echo "Downloading training history..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/checkpoints/${MODEL_BASE}_history.json $LOCAL_DIR/

echo "Downloading validation summary..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/checkpoints/${MODEL_BASE}_validation_summary.csv $LOCAL_DIR/ 2>/dev/null || echo "No validation summary file found."

echo "Downloading validation accuracy..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/checkpoints/${MODEL_BASE}_validation_accuracy.json $LOCAL_DIR/ 2>/dev/null || echo "No validation accuracy file found."

echo "Downloading training curves..."
scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/checkpoints/${MODEL_BASE}_training_curves.png $LOCAL_DIR/ 2>/dev/null || echo "No training curves image found."

# Get the training log file
echo "Checking for training log file..."
LOG_FILE=$(ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && ls -t wav2vec_fixed_training_v4_*.log 2>/dev/null | head -1")
if [ -n "$LOG_FILE" ]; then
  echo "Downloading training log file: $LOG_FILE"
  scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/$LOG_FILE $LOCAL_DIR/
fi

# Checking for normalization files
echo "Checking for normalization files..."
ssh -i $KEY_PATH $EC2_HOST "test -f $REMOTE_DIR/models/wav2vec/wav2vec_mean.npy" && \
  scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/models/wav2vec/wav2vec_mean.npy $LOCAL_DIR/ && \
  echo "Downloaded mean normalization file." || echo "Mean normalization file not found."

ssh -i $KEY_PATH $EC2_HOST "test -f $REMOTE_DIR/models/wav2vec/wav2vec_std.npy" && \
  scp -i $KEY_PATH $EC2_HOST:$REMOTE_DIR/models/wav2vec/wav2vec_std.npy $LOCAL_DIR/ && \
  echo "Downloaded std normalization file." || echo "Std normalization file not found."

echo ""
echo "Download complete! Files saved to $LOCAL_DIR/"
echo ""

# Print validation accuracy if available
if [ -f "$LOCAL_DIR/${MODEL_BASE}_validation_accuracy.json" ]; then
  VAL_ACC=$(cat "$LOCAL_DIR/${MODEL_BASE}_validation_accuracy.json" | grep -o '"val_accuracy": [0-9.]*' | cut -d' ' -f2)
  echo "Model validation accuracy: $VAL_ACC"
fi

# Print instructions for using the model
echo ""
echo "To plot training curves from the history file:"
echo "python scripts/plot_training_curve.py --history_file $LOCAL_DIR/${MODEL_BASE}_history.json --metric both"
