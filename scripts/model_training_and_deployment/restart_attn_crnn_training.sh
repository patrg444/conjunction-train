#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem
export DATASET_PATH="datasets"  # Path to datasets on EC2

echo "=========================================="
echo "Restarting Attention CRNN training on EC2 GPU instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "Dataset path on EC2: $DATASET_PATH"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Clean up any existing failed training sessions
echo "Cleaning up previous training sessions..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "cd ~/emotion_project && pkill -f train_attn_crnn.py || true && tmux kill-session -t attn_train || true"

# Start training in a new tmux session with the correct dataset path
echo "Starting training with the dataset path..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "cd ~/emotion_project && tmux new -s attn_train -d 'source /opt/pytorch/bin/activate && CUDA_VISIBLE_DEVICES=0 python scripts/train_attn_crnn.py --data_dirs $DATASET_PATH --augment > train_log.txt 2>&1'"

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
