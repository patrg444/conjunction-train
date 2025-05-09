#!/bin/bash
# Monitor the progress of the extended wav2vec training on EC2

# Connection details
EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Get the latest log file that matches the pattern
echo "Finding the most recent extended training log file..."
LATEST_LOG=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/wav2vec_extended_training_*.log 2>/dev/null | head -1")

if [ -z "$LATEST_LOG" ]; then
    echo "No extended training log files found. Has the training been started?"
    exit 1
fi

echo "Using log file: $LATEST_LOG"
echo "==============================================================="

# Check if the process is running
echo "Checking if training process is running..."
ssh -i $KEY_PATH $EC2_HOST "ps aux | grep train_wav2vec_extended_epochs.py | grep -v grep || echo 'PROCESS NOT RUNNING!'"

echo ""
echo "Latest log entries:"
echo "==============================================================="
# Get the last 50 lines of the log file
ssh -i $KEY_PATH $EC2_HOST "tail -n 50 $LATEST_LOG"

echo ""
echo "Recent training progress (last 10 epochs):"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -a 'Epoch [0-9][0-9]*/\|val_accuracy' $LATEST_LOG | tail -20"

echo ""
echo "Check for highest validation accuracy achieved so far:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -a 'val_accuracy improved' $LATEST_LOG | tail -5"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
