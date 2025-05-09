#!/bin/bash
# Monitor the training progress of the wav2vec model with neutral-calm mapping

EC2_HOST="ubuntu@54.162.134.77"
KEY_PATH="~/Downloads/gpu-key.pem"
REMOTE_DIR="/home/ubuntu/audio_emotion"

# Find the most recent neutral-calm log file
echo "Finding the most recent neutral-calm training log file..."
LOG_FILE=$(ssh -i $KEY_PATH $EC2_HOST "ls -t $REMOTE_DIR/wav2vec_neutral_calm_*.log 2>/dev/null | head -n 1")

if [ -z "$LOG_FILE" ]; then
    echo "No neutral-calm training log files found!"
    exit 1
fi

echo "Using log file: $LOG_FILE"
echo "==============================================================="

# Check if the training process is still running
echo "Checking if training process is running..."
PROCESS_ID=$(ssh -i $KEY_PATH $EC2_HOST "pgrep -f 'python3 fixed_v5_script_neutral_calm.py'")

if [ -z "$PROCESS_ID" ]; then
    echo "PROCESS NOT RUNNING!"
else
    echo "Process is running with PID: $PROCESS_ID"
    # Get process details
    PROCESS_DETAILS=$(ssh -i $KEY_PATH $EC2_HOST "ps -p $PROCESS_ID -o pid,ppid,user,%cpu,%mem,vsz,rss,stat,start,time,command")
    echo "$PROCESS_DETAILS"
fi

echo ""
echo "Latest log entries:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "tail -n 50 $LOG_FILE"

echo ""
echo "Recent training progress (last 10 epochs):"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep -A 1 'Epoch [0-9]\\+/100' $LOG_FILE | tail -n 20"

echo ""
echo "Check for highest validation accuracy achieved so far:"
echo "==============================================================="
ssh -i $KEY_PATH $EC2_HOST "grep 'val_accuracy improved from' $LOG_FILE"

echo ""
echo "Monitor complete. Run this script again to see updated progress."
