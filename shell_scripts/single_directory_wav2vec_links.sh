#!/bin/bash
# Script to create symlinks only in a single target directory
# This avoids double-counting the WAV2VEC feature files

set -euo pipefail

EC2_IP=$(cat aws_instance_ip.txt)
SSH_KEY="$HOME/Downloads/gpu-key.pem"

echo "=== Creating WAV2VEC Feature Symlinks (Single Directory) ==="
echo "This script will only create 8690 symlinks in /data/wav2vec_features"
echo "without duplicating them in /data/wav2vec_crema_d"

# Connect to EC2 and run the commands
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'EOF'
  # First clean existing symlinks
  echo "Cleaning existing target directories..."
  sudo mkdir -p /data/wav2vec_features
  sudo mkdir -p /data/wav2vec_crema_d
  sudo chown -R ubuntu:ubuntu /data/wav2vec_features
  sudo chown -R ubuntu:ubuntu /data/wav2vec_crema_d
  
  # Remove any existing symlinks
  find /data/wav2vec_features -type l -delete
  find /data/wav2vec_crema_d -type l -delete
  
  # Only create symlinks from the primary WAV2VEC features directory
  PRIMARY_SOURCE="/home/ubuntu/audio_emotion/models/wav2vec"
  
  if [ -d "$PRIMARY_SOURCE" ]; then
    echo "Creating symlinks from $PRIMARY_SOURCE to /data/wav2vec_features only..."
    
    # Count files first
    FILE_COUNT=$(find "$PRIMARY_SOURCE" -name "*.npz" -type f | wc -l)
    echo "Found $FILE_COUNT feature files in primary source directory"
    
    # Create symlinks only to /data/wav2vec_features
    echo "Creating symlinks to /data/wav2vec_features only..."
    find "$PRIMARY_SOURCE" -name "*.npz" -type f | while read file; do
      ln -sf "$file" "/data/wav2vec_features/$(basename "$file")"
    done
    
    # For the training scripts that use /data/wav2vec_crema_d, 
    # create a single symlink pointing to the /data/wav2vec_features directory
    echo "Creating a directory symlink from /data/wav2vec_crema_d to /data/wav2vec_features"
    rmdir /data/wav2vec_crema_d 2>/dev/null || true  # Remove empty directory if it exists
    ln -sfn /data/wav2vec_features /data/wav2vec_crema_d  # Create directory symlink
  else
    echo "ERROR: Primary source directory not found: $PRIMARY_SOURCE"
    exit 1
  fi
  
  # Verify created symlinks
  WAV2VEC_LINKS=$(find /data/wav2vec_features -type l | wc -l)
  
  echo "Created $WAV2VEC_LINKS symlinks in /data/wav2vec_features"
  
  # Verify that symlinks actually resolve to files
  VALID_LINKS_WAV2VEC=$(find -L /data/wav2vec_features -type f | wc -l)
  
  echo "Verified $VALID_LINKS_WAV2VEC valid feature files in /data/wav2vec_features"
  
  if [ "$VALID_LINKS_WAV2VEC" -eq 8690 ]; then
    echo "✓ Exactly 8690 symlinks successfully created and verified"
  else
    echo "✗ Warning: Did not create exactly 8690 symlinks. Please investigate."
  fi

  # Check if the directory symlink is working
  if [ -L "/data/wav2vec_crema_d" ] && [ "$(readlink /data/wav2vec_crema_d)" == "/data/wav2vec_features" ]; then
    echo "✓ Directory symlink from /data/wav2vec_crema_d to /data/wav2vec_features is working"
  else
    echo "✗ Directory symlink from /data/wav2vec_crema_d to /data/wav2vec_features is NOT working"
  fi
EOF

echo
echo "Feature symlink creation complete."
echo "You can now run: ./fixed_stream_attn_crnn_monitor.sh -c"
