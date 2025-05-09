#!/usr/bin/env bash
# Monitoring script for G5 GPU training

SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"
SSH_USER="ubuntu"
AWS_IP="18.208.166.91"
SSH_HOST="$SSH_USER@$AWS_IP"

echo "Connecting to EC2 instance to check training status..."
ssh -i "$SSH_KEY" "$SSH_HOST" "tail -n 50 ~/train_g5_fixed_20250419_132500.log"

# Check GPU usage
echo -e "\nGPU status:"
ssh -i "$SSH_KEY" "$SSH_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv"
