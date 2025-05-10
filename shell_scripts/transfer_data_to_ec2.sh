#!/bin/bash
set -e  # Exit on error

# Configuration
export IP=54.162.134.77
export PEM=~/Downloads/gpu-key.pem
export DATASET_DIR="./datasets"  # Local datasets directory

echo "=========================================="
echo "Transferring datasets to EC2 GPU instance..."
echo "=========================================="
echo "EC2 instance: $IP"
echo "PEM key: $PEM"
echo "Dataset directory: $DATASET_DIR"
echo "=========================================="

# Check if key file exists
if [ ! -f "$PEM" ]; then
    echo "ERROR: PEM file not found at $PEM"
    exit 1
fi

# Fix permissions on key
chmod 600 "$PEM"

# Check if the datasets directory exists locally
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found at $DATASET_DIR"
    exit 1
fi

# Create the target directory on EC2
echo "Creating datasets directory on EC2..."
ssh -i "$PEM" -o StrictHostKeyChecking=no ubuntu@$IP "mkdir -p ~/emotion_project/datasets"

# Transfer datasets to EC2
echo "Transferring datasets (this may take some time)..."
scp -i "$PEM" -o StrictHostKeyChecking=no -r $DATASET_DIR/* ubuntu@$IP:~/emotion_project/datasets/

echo "=========================================="
echo "Dataset transfer complete!"
echo "=========================================="
echo "You can now train the model with:"
echo "./monitor_attn_crnn.sh"
echo "=========================================="
