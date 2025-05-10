#!/bin/bash
set -e

# Variables
KEY=~/Downloads/gpu-key.pem
HOST="ubuntu@$(cat aws_instance_ip.txt)"
REMOTE_DIR="~/conjunction-train"

# Fix shebang line in shell script
sed -i '' '1s/^!/#!/' shell/train_distil_humor.sh

# Ensure target directories exist on remote
ssh -i "$KEY" $HOST "mkdir -p $REMOTE_DIR/datasets/manifests/humor $REMOTE_DIR/configs $REMOTE_DIR/shell $REMOTE_DIR/datasets"

# Upload cleaned CSVs
echo "Uploading cleaned CSV files..."
scp -i "$KEY" \
  datasets/manifests/humor/ur_funny_train_humor_cleaned.csv \
  datasets/manifests/humor/ur_funny_val_humor_cleaned.csv \
  $HOST:$REMOTE_DIR/datasets/manifests/humor/

# Upload YAML config
echo "Uploading config file..."
scp -i "$KEY" \
  configs/train_humor.yaml \
  $HOST:$REMOTE_DIR/configs/

# Upload training shell script
echo "Uploading training script..."
scp -i "$KEY" \
  shell/train_distil_humor.sh \
  $HOST:$REMOTE_DIR/shell/

# Upload the main training script
echo "Uploading Python training script..."
scp -i "$KEY" \
  train_distil_humor.py \
  $HOST:$REMOTE_DIR/

# Upload the dataset module
echo "Uploading dataset module..."
scp -i "$KEY" \
  datasets/fixed_text_dataset.py \
  $HOST:$REMOTE_DIR/datasets/

# Make the shell script executable on remote
ssh -i "$KEY" $HOST "chmod +x $REMOTE_DIR/shell/train_distil_humor.sh"

echo "Upload complete!"
echo "To start training on the EC2 instance, run:"
echo "  ssh -i $KEY $HOST"
echo "  cd ~/conjunction-train"
echo "  bash shell/train_distil_humor.sh --gpus 1 --fp16"
