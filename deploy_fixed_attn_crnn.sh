#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem
export REMOTE_SCRIPT_PATH="~/emotion_project/scripts/fixed_attn_crnn.py"
export DATA_SOURCE="/home/ubuntu/audio_emotion/data"  # Path to existing datasets on EC2

echo "=========================================="
echo "Deploying fixed Attention CRNN script to EC2 GPU instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "Data source on EC2: $DATA_SOURCE"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Clean up any existing sessions
echo "Cleaning up previous training sessions..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "cd ~/emotion_project && pkill -f train_attn_crnn.py || true && tmux kill-session -t attn_train || true"

# Make scripts directory if it doesn't exist
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "mkdir -p ~/emotion_project/scripts"

# Transfer the fixed script
echo "Transferring fixed ATTN-CRNN implementation..."
scp -i "$PEM" fixed_attn_crnn_script.py ubuntu@$IP:$REMOTE_SCRIPT_PATH

# Create models directory
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "mkdir -p ~/emotion_project/models"

# Make sure the script is executable
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "chmod +x $REMOTE_SCRIPT_PATH"

# Start training in a tmux session
echo "Starting training with the fixed script..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "cd ~/emotion_project && tmux new -s attn_train -d 'source /opt/pytorch/bin/activate && CUDA_VISIBLE_DEVICES=0 python scripts/fixed_attn_crnn.py --data_dirs $DATA_SOURCE --augment > train_log.txt 2>&1'"

echo "=========================================="
echo "Training started in tmux session!"
echo "=========================================="
echo "Monitor training progress with:"
echo "  ./monitor_attn_crnn.sh -l  # View logs"
echo "  ./monitor_attn_crnn.sh -s  # Check GPU status"
echo "=========================================="
echo "When training completes, download the model with:"
echo "  ./monitor_attn_crnn.sh -d"
echo "=========================================="
