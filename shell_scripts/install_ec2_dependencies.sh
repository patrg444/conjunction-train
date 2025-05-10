#!/bin/bash
set -e

# Configuration
IP="54.162.134.77"
PEM="$HOME/Downloads/gpu-key.pem"

echo "=== Installing Dependencies on EC2 Instance ==="

ssh -i "$PEM" ubuntu@$IP "
# Install Python packages
echo 'Installing Python packages...'
pip install tensorflow numpy scikit-learn matplotlib seaborn

# Create log directory for TensorBoard
mkdir -p ~/emotion_project/logs

echo 'Dependencies installed successfully!'
"

echo "=== Dependencies Installed Successfully ==="
