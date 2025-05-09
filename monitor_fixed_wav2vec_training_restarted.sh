#!/bin/bash
# Monitor the progress of the wav2vec emotion recognition training with our fixed script
# This script will show the latest training log entries and check for any errors

SSH_KEY="~/Downloads/gpu-key.pem"
REMOTE_HOST="ubuntu@54.162.134.77"
LOG_FILE="/home/ubuntu/audio_emotion/train_wav2vec_audio_only_fixed_v4_restarted.log"

echo "Checking training progress for fixed wav2vec model (restarted)..."
echo "==============================================================="

# Check if the process is running
ssh -i $SSH_KEY $REMOTE_HOST "ps aux | grep train_wav2vec_audio_only_fixed_v4.py | grep -v grep || echo 'PROCESS NOT RUNNING!'"

echo ""
echo "Latest log entries:"
echo "==============================================================="
# Get the last 50 lines of the log file
ssh -i $SSH_KEY $REMOTE_HOST "tail -n 50 $LOG_FILE"

echo ""
echo "Checking for errors in the log file..."
echo "==============================================================="
# Check for common error patterns
ssh -i $SSH_KEY $REMOTE_HOST "grep -i 'error\|exception\|traceback' $LOG_FILE | tail -n 20 || echo 'No errors found.'"

echo ""
echo "Check for ResourceVariable issue (should not appear):"
echo "==============================================================="
# Specifically check for the learning rate ResourceVariable error
ssh -i $SSH_KEY $REMOTE_HOST "grep -i 'ResourceVariable.*not callable' $LOG_FILE || echo 'No ResourceVariable errors found - our fix is working!'"

echo ""
echo "Recent validation accuracy:"
echo "==============================================================="
# Extract validation accuracy numbers
ssh -i $SSH_KEY $REMOTE_HOST "grep -i 'val_accuracy' $LOG_FILE | tail -n 10"

echo ""
echo "Monitor complete. If training is progressing without the ResourceVariable error, our fix is successful."
