#!/bin/bash
# Script to upload facenet feature files and restart training
# This addresses the issue where the training was attempting to run without feature files

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/Users/patrickgloria/Downloads/gpu-key.pem}"
INSTANCE_IP="${1:-18.208.166.91}"
REMOTE_DIR="/home/ubuntu/emotion-recognition"
FEATURES_DIR="crema_d_features_facenet"

if [[ -z "$INSTANCE_IP" ]]; then
  echo "Usage: $0 <instance-ip>"
  echo "Or set INSTANCE_IP environment variable"
  exit 1
fi

echo "=== Uploading Facenet feature files and restarting training ==="
echo "SSH key: $SSH_KEY"
echo "Instance IP: $INSTANCE_IP"
echo "Remote directory: $REMOTE_DIR"
echo "Features directory: $FEATURES_DIR"

# Check if the feature files directory exists
if [[ ! -d "$FEATURES_DIR" ]]; then
  echo "Error: Features directory not found at $FEATURES_DIR"
  exit 1
fi

# Count number of feature files
NUM_FILES=$(find "$FEATURES_DIR" -name "*.npz" | wc -l)
echo "Found $NUM_FILES feature files to upload"

if [[ $NUM_FILES -eq 0 ]]; then
  echo "Error: No .npz files found in $FEATURES_DIR"
  exit 1
fi

# Check SSH connection
echo "Checking SSH connection..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "echo 'SSH connection successful'" || {
  echo "Error: Unable to connect to server. Check your SSH key and server IP."
  exit 1
}

# Stop any existing training
echo "Stopping any existing training processes..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "pkill -f train_facenet_full.py || true"

# Create remote directory
echo "Creating remote directory..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "mkdir -p $REMOTE_DIR/$FEATURES_DIR"

# Upload feature files (using rsync for efficiency)
echo "Uploading feature files (this might take a while)..."
echo "Using rsync to upload files..."

# First create a small batch of files for testing (10 files)
echo "Uploading a sample batch of files first..."
find "$FEATURES_DIR" -name "*.npz" | head -10 | xargs -I{} rsync -avz -e "ssh -i $SSH_KEY" {} ubuntu@"$INSTANCE_IP":"$REMOTE_DIR/$FEATURES_DIR/"

# Check if the sample upload worked
echo "Verifying sample upload..."
SAMPLE_COUNT=$(ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "find $REMOTE_DIR/$FEATURES_DIR -name '*.npz' | wc -l")
echo "Sample files uploaded: $SAMPLE_COUNT"

if [[ $SAMPLE_COUNT -eq 0 ]]; then
  echo "Error: Failed to upload sample files. Check your connection and permissions."
  exit 1
fi

# Continue with the rest of the files
echo "Uploading remaining files..."
rsync -avz -e "ssh -i $SSH_KEY" "$FEATURES_DIR/" ubuntu@"$INSTANCE_IP":"$REMOTE_DIR/$FEATURES_DIR/"

# Verify upload
echo "Verifying upload..."
REMOTE_COUNT=$(ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "find $REMOTE_DIR/$FEATURES_DIR -name '*.npz' | wc -l")
echo "Files uploaded: $REMOTE_COUNT of $NUM_FILES"

if [[ $REMOTE_COUNT -lt 10 ]]; then
  echo "Error: Too few files were uploaded. Something went wrong."
  exit 1
fi

# Restart training
echo "Restarting training in new tmux session..."
ssh -i "$SSH_KEY" ubuntu@"$INSTANCE_IP" "cd $REMOTE_DIR/facenet_full_training && tmux kill-session -t facenet_training || true && tmux new-session -d -s facenet_training './run_full_training.sh > training_output.log 2>&1'"

echo "=== Feature files uploaded and training restarted ==="
echo ""
echo "To monitor training:"
echo "  ./facenet_monitor_helper.sh $INSTANCE_IP"
echo ""
echo "To view live training progress:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo "  tmux attach -t facenet_training"
echo ""
echo "To detach from tmux, press Ctrl+B then D"
