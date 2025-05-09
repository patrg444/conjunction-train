#!/bin/bash
# Script to monitor the fixed WAV2VEC Attention model (v9) training progress

KEY_PATH="~/Downloads/gpu-key.pem"
SERVER="ubuntu@54.162.134.77" 

echo "Finding the most recent fixed v9 training log file..."
SSH_CMD="ls -t /home/ubuntu/audio_emotion/wav2vec_v9_attention_fixed_*.log | head -1"
LATEST_LOG=$(ssh -i $KEY_PATH $SERVER "$SSH_CMD")
echo "Using log file: $LATEST_LOG"
echo "==============================================================="

echo "Checking if training process is running..."
PROCESS_COUNT=$(ssh -i $KEY_PATH $SERVER "ps aux | grep 'python.*fixed_v9_attention_broadcast.py' | grep -v grep | wc -l")
if [ "$PROCESS_COUNT" -gt 0 ]; then
    echo "PROCESS RUNNING (count: $PROCESS_COUNT)"
else
    echo "PROCESS NOT RUNNING!"
fi
echo ""

echo "Latest log entries:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "tail -n 30 $LATEST_LOG"
echo ""

echo "Check for model architecture:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep -A10 'Building model with' $LATEST_LOG | head -12"
echo ""

echo "Check for padding information:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep 'Padding sequences to length' $LATEST_LOG"
ssh -i $KEY_PATH $SERVER "grep 'Padded train shape' $LATEST_LOG"
ssh -i $KEY_PATH $SERVER "grep 'Padded validation shape' $LATEST_LOG"
echo ""

echo "Check for learning rate phases:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep 'CosineDecayWithWarmup setting learning rate' $LATEST_LOG | tail -5"
echo ""

echo "Check for training progress (epochs):"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep -A1 'Epoch [0-9]' $LATEST_LOG | tail -10"
ssh -i $KEY_PATH $SERVER "grep 'val_accuracy' $LATEST_LOG | tail -5"
echo ""

echo "Check for checkpoints saved:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep -A1 'Epoch.*: saving model' $LATEST_LOG | tail -4"
echo ""

echo "Check for any errors:"
echo "==============================================================="
ssh -i $KEY_PATH $SERVER "grep -i error $LATEST_LOG | tail -5"
ssh -i $KEY_PATH $SERVER "grep -i 'invalid_argument' $LATEST_LOG | tail -5"
echo ""

echo "Monitor complete. Run this script again to see updated progress."
