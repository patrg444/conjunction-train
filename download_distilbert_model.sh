#!/bin/bash
set -e

# Get the EC2 instance IP from the file
EC2_IP=$(cat aws_instance_ip.txt)
EC2_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "========================================="
echo "Downloading trained DistilBERT Humor model from EC2 instance ${EC2_IP}"
echo "========================================="

# Create local directories if they don't exist
mkdir -p checkpoints
mkdir -p training_logs_text_humor

# Download the trained model and logs
echo "1. Downloading best model checkpoint..."
scp -i ${EC2_KEY} ubuntu@${EC2_IP}:~/conjunction-train/checkpoints/text_best.ckpt ./checkpoints/

echo "2. Downloading training logs..."
scp -i ${EC2_KEY} ubuntu@${EC2_IP}:~/conjunction-train/training_logs_text_humor/* ./training_logs_text_humor/

echo "========================================="
echo "Download complete! Model available at ./checkpoints/text_best.ckpt"
echo "Training logs saved to ./training_logs_text_humor/"
echo "========================================="
