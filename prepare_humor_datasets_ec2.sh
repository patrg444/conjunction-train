#!/bin/bash
set -e

# Get EC2 instance IP from file or environment variable
if [ -f aws_instance_ip.txt ]; then
    EC2_IP=$(cat aws_instance_ip.txt)
elif [ -n "$EC2_INSTANCE_IP" ]; then
    EC2_IP=$EC2_INSTANCE_IP
else
    echo "Error: EC2 instance IP not found. Please set EC2_INSTANCE_IP or create aws_instance_ip.txt."
    exit 1
fi

# Define EC2 username and the remote directory
EC2_USER="ubuntu"
EC2_PROJECT_DIR="~/humor_detection"
SSH_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "Preparing humor dataset directories on EC2 instance at $EC2_IP..."

# Create the manifests directory on the EC2 instance
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" $EC2_USER@$EC2_IP "mkdir -p $EC2_PROJECT_DIR/datasets/manifests/humor"

# Copy the full UrFunny dataset files to the EC2 instance
echo "Copying local dataset files to EC2 instance..."
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" datasets/manifests/humor/ur_funny_train_humor_cleaned.csv $EC2_USER@$EC2_IP:$EC2_PROJECT_DIR/datasets/manifests/humor/train_humor_with_text.csv
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" datasets/manifests/humor/ur_funny_val_humor_cleaned.csv $EC2_USER@$EC2_IP:$EC2_PROJECT_DIR/datasets/manifests/humor/val_humor_with_text.csv

echo "Dataset preparation complete!"
