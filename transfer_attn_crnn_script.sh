#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem

echo "=========================================="
echo "Transferring Attn-CRNN script to EC2 GPU instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# First check if the script exists locally
if [ ! -f "scripts/train_attn_crnn.py" ]; then
    echo "ERROR: train_attn_crnn.py not found at scripts/train_attn_crnn.py"
    exit 1
fi

# Check if the target directory exists on the remote server
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "mkdir -p ~/emotion_project/scripts"

# Transfer just the single training script file
echo "Transferring train_attn_crnn.py..."
scp -i "$PEM" -o StrictHostKeyChecking=no scripts/train_attn_crnn.py ubuntu@$IP:~/emotion_project/scripts/

echo "=========================================="
echo "Transfer complete!"
echo "=========================================="
echo "You can now connect to the EC2 instance with:"
echo "  ./direct_ec2_connect.sh"
echo ""
echo "Once connected, navigate to the emotion_project directory and run training"
echo "=========================================="
