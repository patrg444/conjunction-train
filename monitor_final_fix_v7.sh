#!/bin/bash
# Monitor training for the final fixed wav2vec model (v7)

# Variables
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
LOG_FILE="wav2vec_final_fix_v7_20250422_142555.log"
REMOTE_DIR="/home/ubuntu/audio_emotion"

echo "Finding the most recent final fix v7 training log file..."
LOG_FILE=$(ssh -i $KEY_PATH $EC2_HOST "find $REMOTE_DIR -name 'wav2vec_final_fix_v7_*.log' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d' '")
echo "Using log file: $LOG_FILE"
echo "==============================================================="

# Check if the process is still running
echo "Checking if training process is running..."
PID=$(ssh -i $KEY_PATH $EC2_HOST "pgrep -f fixed_v7_final_no_weights.py")
if [ -z "$PID" ]; then
    echo "PROCESS NOT RUNNING!"
else
    echo "Process is running with PID $PID"
fi

echo ""
echo "Latest log entries:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "tail -n 50 $LOG_FILE"

# Check for emotion distribution information
echo ""
echo "Check for emotion distribution information:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -A20 'Emotion distribution in dataset' $LOG_FILE | tail -20"

# Check for proper class encoding
echo ""
echo "Check for proper class encoding:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep 'Number of classes after encoding' $LOG_FILE"
ssh -i $KEY_PATH $EC2_HOST "grep 'Original unique label values' $LOG_FILE"

# Check for training progress
echo ""
echo "Check for latest training epoch:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -E 'Epoch [0-9]+/100' $LOG_FILE | tail -5"
ssh -i $KEY_PATH $EC2_HOST "grep -E 'val_accuracy: [0-9.]+' $LOG_FILE | tail -5"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
