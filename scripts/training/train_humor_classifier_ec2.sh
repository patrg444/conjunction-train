#!/bin/bash
# Script to deploy humor classifier training to EC2

# Get the instance IP from file
EC2_IP=$(cat aws_instance_ip.txt)

# Copy manifest files to EC2
echo "Copying manifest files to EC2..."
ssh -i "/Users/patrickgloria/Downloads/gpu-key.pem" ubuntu@$EC2_IP "mkdir -p ~/conjunction-train/datasets/manifests/humor/"
scp -i "/Users/patrickgloria/Downloads/gpu-key.pem" datasets/manifests/humor/combined_train_humor.csv ubuntu@$EC2_IP:~/conjunction-train/datasets/manifests/humor/combined_train_humor.csv
scp -i "/Users/patrickgloria/Downloads/gpu-key.pem" datasets/manifests/humor/combined_val_humor.csv ubuntu@$EC2_IP:~/conjunction-train/datasets/manifests/humor/combined_val_humor.csv

# Run training on EC2
echo "Starting training on EC2..."
ssh -i "/Users/patrickgloria/Downloads/gpu-key.pem" ubuntu@$EC2_IP "cd ~/conjunction-train && \
  python enhanced_train_distil_humor.py \
    --train_manifest datasets/manifests/humor/combined_train_humor.csv \
    --val_manifest datasets/manifests/humor/combined_val_humor.csv \
    --batch_size 64 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --model_name 'distilbert-base-uncased' \
    --output_dir './checkpoints/humor_classifier/' \
    --max_length 128"
