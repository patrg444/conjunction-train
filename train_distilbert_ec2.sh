#!/bin/bash
set -e

# Get the EC2 instance IP from the file
EC2_IP=$(cat aws_instance_ip.txt)
EC2_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "========================================="
echo "Training DistilBERT Humor model on EC2 instance ${EC2_IP}"
echo "========================================="

# Copy necessary files to EC2
echo "Copying training files to EC2 instance..."
echo "1. Copying training script..."
scp -i ${EC2_KEY} enhanced_train_distil_humor.py ubuntu@${EC2_IP}:~/conjunction-train/
echo "2. Setting up directories on EC2..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "mkdir -p ~/conjunction-train/checkpoints ~/conjunction-train/training_logs_text_humor"

# Run the model training on EC2
echo "Starting DistilBERT model training on EC2 instance..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  python enhanced_train_distil_humor.py \
    --train_manifest ~/conjunction-train/datasets/manifests/humor/train_humor_with_text.csv \
    --val_manifest ~/conjunction-train/datasets/manifests/humor/val_humor_with_text.csv \
    --model_name distilbert-base-uncased \
    --max_length 128 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_workers 4 \
    --log_dir ~/conjunction-train/training_logs_text_humor"

echo "========================================="
echo "Training complete! Best model saved at ~/conjunction-train/checkpoints/text_best.ckpt on EC2"
echo "========================================="
