#!/bin/bash
# Monitor the training progress of the wav2vec model with fixed continuous emotion indices

# Define variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"
MODEL_PREFIX="wav2vec_six_classes"

# Find the most recent log file that matches our model name
echo "Finding the most recent six-classes training log file..."
LOG_FILE=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/wav2vec_six_classes_*.log 2>/dev/null | head -n 1")

if [ -z "$LOG_FILE" ]; then
  echo "No training log found for ${MODEL_PREFIX}!"
  exit 1
fi

echo "Using log file: $LOG_FILE"
echo "==============================================================="

# Check if the training process is still running
echo "Checking if training process is running..."
PROCESS_RUNNING=$(ssh -i $KEY_PATH $EC2_HOST "ps aux | grep 'python3 fixed_v5_script_continuous_indices.py' | grep -v grep")
if [ -z "$PROCESS_RUNNING" ]; then
  echo "PROCESS NOT RUNNING!"
else
  echo "PROCESS IS RUNNING"
fi

echo ""
echo "Latest log entries:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "tail -n 50 $LOG_FILE"

# Find highest validation accuracy and epoch
echo ""
echo "Check for highest validation accuracy achieved so far:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep 'val_accuracy improved from' $LOG_FILE"

# Extract epoch distribution information to confirm proper class mapping
echo ""
echo "Emotion distribution information (check for 6 classes and proper mapping):"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -A 15 'Using emotion mapping:' $LOG_FILE | tail -15"
ssh -i $KEY_PATH $EC2_HOST "grep -A 15 'Class distribution after encoding:' $LOG_FILE | tail -15"
ssh -i $KEY_PATH $EC2_HOST "grep 'Number of classes after encoding:' $LOG_FILE"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
