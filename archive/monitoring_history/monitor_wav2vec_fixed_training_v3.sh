#!/bin/bash
# Enhanced monitoring script for wav2vec emotion recognition training (v3)
# This version has better error detection and displays training stats

set -e  # Exit on error

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Get the latest log file with glob pattern matching
echo "Finding latest training log file..."
LOG_FILE=$(ssh -i $KEY_PATH $EC2_HOST "cd $REMOTE_DIR && ls -t wav2vec_fixed_training_v3_*.log 2>/dev/null | head -1")

if [ -z "$LOG_FILE" ]; then
  echo "No log file found matching pattern 'wav2vec_fixed_training_v3_*.log'"
  LOG_FILE="wav2vec_fixed_training_v3_$(date +"%Y%m%d")"_*.log
  echo "Will use default pattern: $LOG_FILE"
fi

echo "Enhanced Wav2Vec Training Monitor (v3)"
echo "================================================"
echo "Monitoring log file: $REMOTE_DIR/$LOG_FILE"

# Check if the training process is running
TRAINING_PID=$(ssh -i $KEY_PATH $EC2_HOST "pgrep -f 'python3.*train_wav2vec_audio_only_fixed_v3.py' || echo ''")

if [ -n "$TRAINING_PID" ]; then
  echo "[✓] Training process is running with PID $TRAINING_PID"
else
  echo "[✗] Training process NOT FOUND"
  echo "  - Check for error messages in log file"
  echo "  - The process may have completed or crashed"
fi

# Display GPU stats
echo ""
echo "GPU Stats:"
ssh -i $KEY_PATH $EC2_HOST "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits" | awk -F, '{print $1 " % " $2 " % " $3 " MiB " $4 " MiB " $5}'

# Check log file for key information
echo ""
echo "Log Analysis:"

# Check for NaN values in training
NAN_VALUES=$(ssh -i $KEY_PATH $EC2_HOST "grep -i 'nan' $REMOTE_DIR/$LOG_FILE | grep -i 'loss' || echo ''")
if [ -n "$NAN_VALUES" ]; then
  echo "[!] NaN values detected in training:"
  echo "$NAN_VALUES" | head -3
fi

# Check if NPZ files were found successfully
VALID_FILES_COUNT=$(ssh -i $KEY_PATH $EC2_HOST "grep 'Using [0-9]* valid files' $REMOTE_DIR/$LOG_FILE | tail -1 || echo ''")
if [ -n "$VALID_FILES_COUNT" ]; then
  echo "[✓] $VALID_FILES_COUNT"
else
  echo "[?] Could not determine how many valid files are being used"
fi

# Check for error messages
ERROR_MESSAGES=$(ssh -i $KEY_PATH $EC2_HOST "grep -i 'error\|exception\|failed\|no valid files' $REMOTE_DIR/$LOG_FILE | tail -5 || echo ''")
if [ -n "$ERROR_MESSAGES" ]; then
  echo "[!] Errors detected in training:"
  echo "$ERROR_MESSAGES"
fi

# Get recent skipped files info
SKIPPED_FILES=$(ssh -i $KEY_PATH $EC2_HOST "grep 'Skipped .* files due to parsing' $REMOTE_DIR/$LOG_FILE | tail -1 || echo ''")
if [ -n "$SKIPPED_FILES" ]; then
  echo "$SKIPPED_FILES"
fi

# Extract any normalization stats info
NORM_STATS=$(ssh -i $KEY_PATH $EC2_HOST "grep 'normalization statistics' $REMOTE_DIR/$LOG_FILE | tail -2 || echo ''")
if [ -n "$NORM_STATS" ]; then
  echo "$NORM_STATS"
fi

# Get training progress
echo ""
echo "Recent loss values:"
ssh -i $KEY_PATH $EC2_HOST "grep -A 1 'Epoch' $REMOTE_DIR/$LOG_FILE | grep 'loss' | tail -5 || echo ''"

echo ""
echo "Recent accuracy values:"
ssh -i $KEY_PATH $EC2_HOST "grep -A 1 'Epoch' $REMOTE_DIR/$LOG_FILE | grep 'acc' | tail -5 || echo ''"

# Check learning rate
echo ""
echo "Current learning rate:"
ssh -i $KEY_PATH $EC2_HOST "grep 'learning rate' $REMOTE_DIR/$LOG_FILE | tail -1 || echo ''"

# Check TensorBoard status
TB_RUNNING=$(ssh -i $KEY_PATH $EC2_HOST "pgrep -f tensorboard || echo ''")
if [ -n "$TB_RUNNING" ]; then
  echo ""
  echo "[✓] TensorBoard is running"
  echo "To view TensorBoard run:"
  echo "ssh -i $KEY_PATH -L 6006:localhost:6006 $EC2_HOST"
  echo "Then open http://localhost:6006 in your browser"
else
  echo ""
  echo "[i] TensorBoard is not running"
  echo "To start TensorBoard run:"
  echo "ssh -i $KEY_PATH $EC2_HOST \"cd $REMOTE_DIR && tensorboard --logdir=logs\""
  echo "In a separate terminal, set up a tunnel:"
  echo "ssh -i $KEY_PATH -L 6006:localhost:6006 $EC2_HOST"
  echo "Then open http://localhost:6006 in your browser"
fi

echo ""
echo "To tail the log file continuously run:"
echo "ssh -i $KEY_PATH $EC2_HOST \"tail -f $REMOTE_DIR/$LOG_FILE\""

echo ""
echo "Monitoring complete"
