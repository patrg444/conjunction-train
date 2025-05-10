#!/bin/bash
set -e

# Get the EC2 instance IP from the file
EC2_IP=$(cat aws_instance_ip.txt)
EC2_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "========================================="
echo "Training DistilBERT on Talk-Level Humor dataset on EC2 instance ${EC2_IP}"
echo "========================================="

# Copy necessary files to EC2
echo "Copying training files to EC2 instance..."
echo "1. Copying training script..."
scp -i ${EC2_KEY} enhanced_train_distil_humor.py ubuntu@${EC2_IP}:~/conjunction-train/
echo "2. Setting up directories on EC2..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "mkdir -p ~/conjunction-train/checkpoints ~/conjunction-train/training_logs_talk_level_humor"

# Run the model training on EC2
echo "Starting DistilBERT model training on talk-level data on EC2 instance..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  python enhanced_train_distil_humor.py \
    --train_manifest ~/conjunction-train/datasets/manifests/humor/talk_level_train_humor.csv \
    --val_manifest ~/conjunction-train/datasets/manifests/humor/talk_level_val_humor.csv \
    --model_name distilbert-base-uncased \
    --max_length 256 \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_workers 4 \
    --log_dir ~/conjunction-train/training_logs_talk_level_humor \
    --model_save_path ~/conjunction-train/checkpoints/talk_level_humor_model.ckpt"

echo "========================================="
echo "Training complete! Best model saved at ~/conjunction-train/checkpoints/talk_level_humor_model.ckpt on EC2"
echo "========================================="

# Download the model from EC2
echo "Downloading trained model from EC2..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  tar -czvf talk_level_humor_model.tar.gz checkpoints/talk_level_humor_model.ckpt"

mkdir -p ./checkpoints
scp -i ${EC2_KEY} ubuntu@${EC2_IP}:~/conjunction-train/talk_level_humor_model.tar.gz ./checkpoints/
tar -xzvf ./checkpoints/talk_level_humor_model.tar.gz -C ./
rm ./checkpoints/talk_level_humor_model.tar.gz

echo "========================================="
echo "Model downloaded and extracted to ./checkpoints/talk_level_humor_model.ckpt"
echo "========================================="
