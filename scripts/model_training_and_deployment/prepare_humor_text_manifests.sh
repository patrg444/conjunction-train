#!/bin/bash
set -e

# Get the EC2 instance IP from the file
EC2_IP=$(cat aws_instance_ip.txt)
EC2_KEY="/Users/patrickgloria/Downloads/gpu-key.pem"

echo "========================================="
echo "Preparing humor manifests with text on EC2 instance ${EC2_IP}"
echo "========================================="

# Step 1: Generate base humor manifests from SMILE dataset
echo "Step 1: Generating base humor manifests from SMILE dataset..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  mkdir -p datasets/manifests/humor && \
  python scripts/prepare_humor_manifests.py \
    --output_dir datasets/manifests/humor \
    --dataset_root /home/ubuntu/datasets \
    --smile_dir SMILE"

# Step 2: Split the humor manifest into train and val
echo "Step 2: Splitting humor manifests into train and val..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  python -c \"
import pandas as pd
import os
# Read the humor manifest
humor_df = pd.read_csv('datasets/manifests/humor/humor_manifest.csv')
# Shuffle the data
humor_df = humor_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Split into train (80%) and val (20%)
train_size = int(0.8 * len(humor_df))
train_df = humor_df[:train_size]
val_df = humor_df[train_size:]
# Save to files
train_df.to_csv('datasets/manifests/humor/train_humor.csv', index=False)
val_df.to_csv('datasets/manifests/humor/val_humor.csv', index=False)
print(f'Split {len(humor_df)} entries into {len(train_df)} train and {len(val_df)} val')
\""

# Step 3: Add text transcripts to the humor manifests
echo "Step 3: Adding text transcripts to the humor manifests..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  python scripts/augment_humor_manifest_with_text.py"

# Step A fix paths in augment_humor_manifest_with_text.py if needed
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  sed -i 's|datasets/humor_train.csv|datasets/manifests/humor/train_humor.csv|g' scripts/augment_humor_manifest_with_text.py && \
  sed -i 's|datasets/humor_val.csv|datasets/manifests/humor/val_humor.csv|g' scripts/augment_humor_manifest_with_text.py && \
  sed -i 's|datasets/humor_train_with_text.csv|datasets/manifests/humor/train_humor_with_text.csv|g' scripts/augment_humor_manifest_with_text.py && \
  sed -i 's|datasets/humor_val_with_text.csv|datasets/manifests/humor/val_humor_with_text.csv|g' scripts/augment_humor_manifest_with_text.py"

# Step 3 (retry): Add text transcripts to the humor manifests
echo "Step 3 (retry): Adding text transcripts to the humor manifests..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  python scripts/augment_humor_manifest_with_text.py"

# Step 4: Verify the manifests with text exist
echo "Step 4: Verifying the manifests with text exist..."
ssh -i ${EC2_KEY} ubuntu@${EC2_IP} "cd ~/conjunction-train && \
  ls -la datasets/manifests/humor/ && \
  head -n 3 datasets/manifests/humor/train_humor_with_text.csv"

echo "========================================="
echo "Humor manifests with text prepared!"
echo "========================================="
